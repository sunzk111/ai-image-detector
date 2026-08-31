"""Build 100k train + 4k internal validation; never train on COCO/DALL-E.

Colab: python -u prepare_mixed_dataset.py --base-config config.yaml
No network: --dry-run. Resume: run the same command and keep data/mixed100k_v2.
The existing config.mixed.yaml (90k model evaluation) is never overwritten.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import csv
import hashlib
import io
import json
import os
import random
import re
import shutil
import sqlite3
import struct
import subprocess
import threading
import time
import zipfile
import zlib
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import quote

VERSION = 3
REPO = "hy2628982280/WildFake"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
FIELDS = ["path", "label", "split", "generator", "id", "source_dataset",
          "usage", "source_key", "pixel_sha256", "file_sha256"]


@dataclass(frozen=True)
class Source:
    name: str
    dataset: str
    generator: str
    label: int
    train: int
    val: int
    archive: str = ""
    upstream_split: str = ""

    @property
    def total(self):
        return self.train + self.val


SOURCES = (
    Source("cifake_real", "cifake", "CIFAKE_real", 0, 5000, 0, upstream_split="train"),
    Source("cifake_fake", "cifake", "CIFAKE_SD14", 1, 5000, 0, upstream_split="train"),
    Source("cifake_real_val", "cifake", "CIFAKE_real", 0, 0, 250, upstream_split="test"),
    Source("cifake_fake_val", "cifake", "CIFAKE_SD14", 1, 0, 250, upstream_split="test"),
    Source("sid_real", "sid_set", "SID_real", 0, 5000, 0, upstream_split="train"),
    Source("sid_fake", "sid_set", "SID_full_synthetic", 1, 5000, 0, upstream_split="train"),
    Source("sid_real_val", "sid_set", "SID_real", 0, 0, 250, upstream_split="validation"),
    Source("sid_fake_val", "sid_set", "SID_full_synthetic", 1, 0, 250, upstream_split="validation"),
    Source("wf_imagenet", "wildfake", "ImageNet", 0, 20000, 750, "Images/Real/imagenet.zip"),
    Source("wf_church", "wildfake", "Church", 0, 6000, 225, "Images/Real/church.zip"),
    Source("wf_afhq", "wildfake", "AFHQ", 0, 6000, 225, "Images/Real/afhq.zip"),
    Source("wf_ffhq", "wildfake", "FFHQ", 0, 4000, 150, "Images/Real/ffhq.zip"),
    Source("wf_celebahq", "wildfake", "CelebA-HQ", 0, 4000, 150, "Images/Real/celebahq.zip"),
    Source("wf_ddim", "wildfake", "DDIM", 1, 8000, 300, "Images/Diffusion_based/DDIM.zip"),
    Source("wf_ddpm", "wildfake", "DDPM", 1, 8000, 300, "Images/Diffusion_based/DDPM.zip"),
    Source("wf_adm", "wildfake", "ADM", 1, 8000, 300, "Images/Diffusion_based/ADM.zip"),
    Source("wf_imagen", "wildfake", "Imagen", 1, 8000, 300, "Images/Diffusion_based/Imagen.zip"),
    Source("wf_vqdm", "wildfake", "VQDM", 1, 8000, 300, "Images/Diffusion_based/VQDM.zip"),
)
ALLOWED_ARCHIVES = frozenset(s.archive for s in SOURCES if s.archive)


def log(message):
    print(message, flush=True)


def forbidden(key):
    key = str(key).replace("\\", "/").lower()
    return bool(re.search(r"(^|[/_.-])(coco|val2017|dall[-_]?e\d*)([/_.-]|$)", key))


def image_paths(root):
    for directory, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d != "__MACOSX")
        for name in sorted(files):
            if Path(name).suffix.lower() in EXTENSIONS and not name.startswith("._"):
                yield Path(directory) / name


def seeded_order(items, seed, name):
    result = sorted(items, key=lambda x: x.filename if isinstance(x, zipfile.ZipInfo) else str(x))
    number = int(hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()[:16], 16)
    random.Random(number).shuffle(result)
    return result


def image_info(raw):
    from PIL import Image
    with Image.open(io.BytesIO(raw)) as im:
        fmt = im.format
        im.load()  # Reject truncation now rather than during GPU training.
        rgb = im.convert("RGB")
        digest = hashlib.sha256(struct.pack("<II", *rgb.size) + rgb.tobytes()).hexdigest()
    suffix = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "BMP": ".bmp",
              "TIFF": ".tiff"}.get(fmt)
    if suffix is None:
        raise ValueError(f"Unsupported image format: {fmt}")
    return digest, suffix


def atomic_bytes(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(raw)
    temporary.replace(path)


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode())


def test_fingerprints(test_root, cache):
    """Only exact-pixel overlap auditing, never scores/threshold fitting on tests."""
    if not test_root.is_dir():
        log("Test root absent: archive/path exclusion active; exact test-overlap audit unavailable.")
        return set(), 0
    old = json.loads(cache.read_text()) if cache.exists() else {}
    current = {}
    files = list(image_paths(test_root))
    log(f"Auditing {len(files)} existing test images for exact-pixel overlap (no downloading).")
    for n, path in enumerate(files, 1):
        st = path.stat()
        key = str(path.resolve())
        stamp = [st.st_size, st.st_mtime_ns]
        record = old.get(key)
        if not record or record["stamp"] != stamp:
            digest, _ = image_info(path.read_bytes())
            record = {"stamp": stamp, "digest": digest}
        current[key] = record
        if n % 1000 == 0:
            log(f"  test overlap index {n}/{len(files)}")
            atomic_json(cache, current)
    atomic_json(cache, current)
    return {v["digest"] for v in current.values()}, len(files)


class State:
    def __init__(self, project, root, seed, excluded=(), sources=SOURCES):
        self.project, self.root, self.seed = project.resolve(), root.resolve(), seed
        self.sources = {s.name: s for s in sources}
        root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(root / "state.sqlite3")
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS samples (
                source TEXT, source_key TEXT, path TEXT UNIQUE, label INTEGER,
                split TEXT, generator TEXT, source_dataset TEXT,
                pixel_sha256 TEXT UNIQUE, file_sha256 TEXT, size INTEGER,
                PRIMARY KEY(source, source_key));
        """)
        signature = json.dumps({"version": VERSION, "seed": seed,
                                "sources": [asdict(s) for s in sources]}, sort_keys=True)
        previous = self.db.execute("SELECT value FROM metadata WHERE key='plan'").fetchone()
        if previous and previous[0] != signature:
            self.db.close()
            raise RuntimeError("Cached plan/seed differs. Restore its plan or use a new --data-dir.")
        self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('plan', ?)", (signature,))
        self.excluded = set(excluded)
        self.rows, self.keys, self.digests = [], set(), set()
        self.counts = Counter()
        self.rejected = Counter()
        for record in self.db.execute("SELECT * FROM samples ORDER BY rowid").fetchall():
            row = dict(record)
            path = self.project / row["path"]
            if not path.is_file() or path.stat().st_size != row["size"] or row["pixel_sha256"] in self.excluded:
                self.db.execute("DELETE FROM samples WHERE source=? AND source_key=?",
                                (row["source"], row["source_key"]))
                continue
            self._remember(row)
        self.db.commit()

    def _remember(self, row):
        self.rows.append(row)
        self.keys.add((row["source"], row["source_key"]))
        self.digests.add(row["pixel_sha256"])
        self.counts[row["source"], row["split"]] += 1

    def count(self, source):
        return self.counts[source.name, "train"] + self.counts[source.name, "val"]

    def complete(self, source):
        return self.count(source) == source.total

    def accept(self, source, key, raw):
        if self.complete(source) or (source.name, key) in self.keys:
            return False
        if forbidden(key):
            self.rejected["forbidden_path"] += 1
            return False
        try:
            digest, suffix = image_info(raw)
        except Exception as exc:
            self.rejected["unreadable"] += 1
            log(f"  skip unreadable {source.name}/{key}: {type(exc).__name__}")
            return False
        if digest in self.excluded or digest in self.digests:
            self.rejected["test_overlap" if digest in self.excluded else "duplicate"] += 1
            return False
        split = "val" if self.counts[source.name, "val"] < source.val else "train"
        file_id = hashlib.sha256(key.encode()).hexdigest()[:24]
        path = self.root / "images" / source.name / (file_id + suffix)
        atomic_bytes(path, raw)  # Keep original encoded image; do not introduce JPEG/source shortcuts.
        row = dict(source=source.name, source_key=key, path=path.relative_to(self.project).as_posix(),
                   label=source.label, split=split, generator=source.generator,
                   source_dataset=source.dataset, pixel_sha256=digest,
                   file_sha256=hashlib.sha256(raw).hexdigest(), size=len(raw))
        columns = ",".join(row)
        self.db.execute(f"INSERT INTO samples ({columns}) VALUES ({','.join('?' for _ in row)})", tuple(row.values()))
        self._remember(row)
        n = self.count(source)
        if n % 50 == 0 or self.complete(source):
            self.db.commit()
        if n % 250 == 0 or self.complete(source):
            log(f"  {source.name}: {n}/{source.total} | all accepted {len(self.rows)}/104000")
        return True

    def require_complete(self, source):
        if not self.complete(source):
            raise RuntimeError(f"{source.name}: only {self.count(source)}/{source.total} usable unique images. "
                               "No automatic substitution/oversampling. Completed samples are preserved.")

    def close(self):
        self.db.commit()
        self.db.close()


class RangeHTTP:
    """Strict ranged reads; never silently fetch a multi-GB file if Range is ignored."""
    def __init__(self, archive, session=None):
        if archive not in ALLOWED_ARCHIVES:
            raise ValueError(f"Archive is not in the training allowlist: {archive}")
        if session is None:
            import requests
            session = requests.Session()
        self.session = session
        self.origin = (f"https://modelscope.cn/api/v1/datasets/{REPO}/repo?Revision=master&FilePath="
                       + quote(archive, safe=""))
        self.url, self.size = self.origin, None
        self.get(0, 1)

    def get(self, offset, length):
        if length <= 0:
            return b""
        if length > 64 * 1024 * 1024:
            raise RuntimeError("Refusing an unexpectedly large range read (>64 MiB).")
        end = offset + length - 1
        if self.size is not None:
            end = min(end, self.size - 1)
        for attempt in range(5):
            try:
                with self.session.get(self.url, headers={"Range": f"bytes={offset}-{end}",
                                                        "Accept-Encoding": "identity"},
                                      stream=True, timeout=(20, 90)) as response:
                    if response.status_code in (401, 403) and self.url != self.origin:
                        self.url = self.origin  # Refresh an expired CDN redirect, not authentication.
                        continue
                    response.raise_for_status()
                    if response.status_code != 206:
                        raise ValueError("Server did not honor HTTP Range; use --download-mode full.")
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if not match or int(match[1]) != offset or int(match[2]) != end:
                        raise ValueError("Incorrect Content-Range; response refused.")
                    total = int(match[3])
                    if self.size is not None and self.size != total:
                        raise ValueError("Archive size changed during download.")
                    raw = response.content
                    if len(raw) != end - offset + 1:
                        raise IOError("Incomplete HTTP range response")
                    self.url, self.size = response.url, total
                    return raw
            except ValueError:
                raise
            except Exception as exc:
                if attempt == 4:
                    raise RuntimeError(f"Range read failed ({type(exc).__name__}); rerun to resume.") from exc
                log(f"  network retry {attempt + 1}/4 ({type(exc).__name__})")
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError("CDN access could not be refreshed; rerun or use --download-mode full.")


class RangeReader(io.RawIOBase):
    def __init__(self, http):
        self.http, self.position = http, 0

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.http.size + offset
        if self.position < 0:
            raise ValueError("Negative archive offset")
        return self.position

    def read(self, size=-1):
        size = self.http.size - self.position if size < 0 else min(size, self.http.size - self.position)
        raw = self.http.get(self.position, max(0, size))
        self.position += len(raw)
        return raw


def member_bytes(http, info):
    if info.flag_bits & 1:
        raise ValueError("Encrypted ZIP members are unsupported")
    raw = http.get(info.header_offset, 30 + 1024 + info.compress_size)
    if raw[:4] != b"PK\x03\x04" or len(raw) < 30:
        raise ValueError("Invalid ZIP local header")
    name_len, extra_len = struct.unpack_from("<HH", raw, 26)
    start = 30 + name_len + extra_len
    if len(raw) < start + info.compress_size:
        raw = http.get(info.header_offset, start + info.compress_size)
    compressed = raw[start:start + info.compress_size]
    if info.compress_type == zipfile.ZIP_STORED:
        data = compressed
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(compressed, -15)
    else:
        raise ValueError("Unsupported remote ZIP method; use --download-mode full.")
    if len(data) != info.file_size or zlib.crc32(data) & 0xffffffff != info.CRC:
        raise ValueError("ZIP member length/CRC mismatch")
    return data


def valid_members(zf):
    return [i for i in zf.infolist() if not i.is_dir()
            and Path(i.filename).suffix.lower() in EXTENSIONS
            and "__MACOSX" not in i.filename and not Path(i.filename).name.startswith("._")
            and not forbidden(i.filename)]


def prepare_wildfake(state, source, archive_dir, mode, workers):
    if state.complete(source):
        log(f"Ready: {source.name} ({source.total})")
        return
    if source.archive not in ALLOWED_ARCHIVES:
        raise ValueError("Disallowed archive")
    local = archive_dir / source.archive
    full = local.is_file() or mode == "full" or source.label == 0
    log(f"WildFake {source.generator}: {source.train} train + {source.val} internal val; "
        f"{'local/full ZIP' if full else 'parallel selective Range download'}")
    if full and not local.is_file():
        cli = shutil.which("ms-hub")
        if cli is None:
            raise RuntimeError("Missing ms-hub. Install modelscope-hub in the preparation cell.")
        environment = dict(os.environ, MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS=str(workers))
        subprocess.run([cli, "download", REPO, source.archive, "--repo-type", "dataset",
                        "--local-dir", str(archive_dir), "--max-workers", str(workers)],
                       check=True, env=environment)
    if full:
        with zipfile.ZipFile(local) as zf:
            members = seeded_order(valid_members(zf), state.seed, source.name)
            for info in members:
                if state.complete(source):
                    break
                if (source.name, info.filename) in state.keys:
                    continue
                # CRC errors are fatal and actionable, not silently replaced with missing data.
                state.accept(source, info.filename, zf.read(info))
    else:
        http = RangeHTTP(source.archive)
        with zipfile.ZipFile(RangeReader(http)) as zf:
            members = seeded_order(valid_members(zf), state.seed, source.name)
        log(f"  archive contains {len(members)} allowed images; full size {http.size / 1e9:.2f} GB")
        members = [i for i in members if (source.name, i.filename) not in state.keys]
        thread_state = threading.local()

        def fetch(info):
            if not hasattr(thread_state, "http"):
                thread_state.http = RangeHTTP(source.archive)
            return info.filename, member_bytes(thread_state.http, info)

        with cf.ThreadPoolExecutor(max_workers=workers) as pool:
            cursor = 0
            while cursor < len(members) and not state.complete(source):
                n = min(workers * 2, source.total - state.count(source))
                batch = members[cursor:cursor + n]
                cursor += len(batch)
                # Ordered map preserves source selection/splits despite network completion order.
                for key, raw in pool.map(fetch, batch):
                    state.accept(source, key, raw)
    state.require_complete(source)


def find_cifake(root):
    if (root / "train" / "REAL").is_dir():
        return root
    for path in sorted(root.iterdir()):
        if path.is_dir() and (path / "train" / "REAL").is_dir():
            return path
    raise FileNotFoundError(f"CIFAKE train/REAL folder not found under {root}")


def prepare_cifake(state, root):
    sources = [s for s in SOURCES if s.dataset == "cifake"]
    if all(state.complete(s) for s in sources):
        return
    if root is None:
        import kagglehub
        root = Path(kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images"))
    root = find_cifake(root)
    for source in sources:
        if state.complete(source):
            continue
        folder = root / source.upstream_split / ("REAL" if source.label == 0 else "FAKE")
        log(f"CIFAKE {source.name}: copying only {source.total} selected images to local training storage.")
        for path in seeded_order(list(image_paths(folder)), state.seed, source.name):
            if state.complete(source):
                break
            key = path.relative_to(root).as_posix()
            if (source.name, key) not in state.keys:
                state.accept(source, key, path.read_bytes())
        state.require_complete(source)


def prepare_sid(state):
    from datasets import Image as HFImage, load_dataset
    for split in ("train", "validation"):
        sources = {s.label: s for s in SOURCES if s.dataset == "sid_set" and s.upstream_split == split}
        if all(state.complete(s) for s in sources.values()):
            continue
        log(f"SID_Set {split}: streaming label 0/1 only; mask and label 2 excluded.")
        stream = load_dataset("saberzl/SID_Set", split=split, streaming=True)
        stream = stream.select_columns(["img_id", "image", "label"]).cast_column("image", HFImage(decode=False))
        stream = stream.filter(lambda r: int(r["label"]) in (0, 1)).shuffle(seed=state.seed, buffer_size=512)
        for row in stream:
            source = sources[int(row["label"])]
            if state.complete(source):
                continue
            key = f"{split}/{row['img_id']}"
            if (source.name, key) in state.keys:
                continue
            feature = row["image"]
            raw = feature.get("bytes")
            if raw is None:
                raw = Path(feature["path"]).read_bytes()
            state.accept(source, key, raw)
            if all(state.complete(s) for s in sources.values()):
                break
        for source in sources.values():
            state.require_complete(source)


def manifest_rows(state):
    result = []
    for row in state.rows:
        source = state.sources[row["source"]]
        if forbidden(row["source_key"]) or source.archive and source.archive not in ALLOWED_ARCHIVES:
            raise ValueError("Prohibited provenance detected")
        result.append({key: value for key, value in dict(
            row, id=row["source"] + ":" + hashlib.sha256(row["source_key"].encode()).hexdigest(),
            usage="training" if row["split"] == "train" else "internal_validation"
        ).items() if key in FIELDS})
    return result


def validate_rows(rows, sources=SOURCES):
    expected = Counter()
    for source in sources:
        expected["train", source.dataset, source.label] += source.train
        expected["val", source.dataset, source.label] += source.val
    expected = +expected
    actual = Counter((r["split"], r["source_dataset"], int(r["label"])) for r in rows)
    if actual != expected:
        raise ValueError(f"Quota mismatch: expected={expected}, actual={actual}")
    for key in ("id", "path", "pixel_sha256"):
        if len({r[key] for r in rows}) != len(rows):
            raise ValueError(f"Duplicate {key} across train/val")
    if any(forbidden(r["source_key"]) or r["usage"] not in ("training", "internal_validation") for r in rows):
        raise ValueError("Forbidden test data in manifest")


def write_csv(path, rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(path, buffer.getvalue().encode("utf-8"))


def write_outputs(state, base_config, test_root, test_count):
    import yaml
    for source in SOURCES:
        state.require_complete(source)
    rows = manifest_rows(state)
    validate_rows(rows)
    random.Random(state.seed).shuffle(rows)
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    for name, subset in (("train.csv", train), ("val.csv", val)):
        write_csv(state.root / name, subset)
    smoke = []
    smoke_val = []
    for dataset in ("cifake", "sid_set", "wildfake"):
        for label in (0, 1):
            smoke.extend([r for r in train if r["source_dataset"] == dataset and r["label"] == label][:100])
            smoke_val.extend([r for r in val if r["source_dataset"] == dataset and r["label"] == label][:50])
    write_csv(state.root / "train_smoke.csv", smoke)
    write_csv(state.root / "val_smoke.csv", smoke_val)

    config = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    config["output_dir"] = "outputs/dinov2_mixed100k_v2"
    prefix = state.root.relative_to(state.project).as_posix()

    def src(name, split):
        result = {"kind": "manifest", "path": f"{prefix}/{name}.csv", "root": ".", "split": split}
        if split == "train":
            result["forbid"] = {"usage": ["validation_only", "test_only", "internal_validation"]}
        return result

    data = config.setdefault("data", {})
    data["train"], data["val"] = src("train", "train"), src("val", "val")
    data["label_map"] = {0: 0, 1: 1}
    data.setdefault("class_to_label", {}).update({"coco": 0, "dalle": 1})
    data["validation_demo"] = {"kind": "imagefolder", "path": str(test_root)}
    config.setdefault("training", {})["resume"] = None
    atomic_bytes(state.project / "config.mixed100k.yaml", yaml.safe_dump(config, sort_keys=False).encode())
    config = copy.deepcopy(config)
    config["output_dir"] = "outputs/smoke_mixed100k_v2"
    config["data"]["train"], config["data"]["val"] = src("train_smoke", "train"), src("val_smoke", "val")
    config["training"]["epochs"] = 1
    atomic_bytes(state.project / "config.mixed100k.smoke.yaml", yaml.safe_dump(config, sort_keys=False).encode())
    atomic_json(state.root / "summary.json", {
        "version": VERSION, "seed": state.seed, "train": len(train), "internal_val": len(val),
        "quotas": [asdict(s) for s in SOURCES], "rejected": dict(state.rejected),
        "test_images_indexed": test_count, "overlap_check": "decoded RGB exact pixels (not perceptual similarity)",
        "test_used_for_training_or_model_selection": False,
    })
    log("READY: 100000 train (50000 real + 50000 AI), 4000 internal validation (2000 + 2000).")
    log("Created config.mixed100k.yaml and config.mixed100k.smoke.yaml; existing config.mixed.yaml unchanged.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--data-dir", type=Path, default=Path("data/mixed100k_v2"))
    parser.add_argument("--base-config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--test-root", type=Path, default=Path("data/wildfake_eval"))
    parser.add_argument("--archive-dir", type=Path, default=Path("data/wildfake_raw"))
    parser.add_argument("--cifake-root", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download-mode", choices=("selective", "full"), default="selective")
    parser.add_argument("--existing-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("--workers must be between 1 and 16")
    if args.dry_run:
        for s in SOURCES:
            log(f"{s.name:20} label={s.label} train={s.train:5} val={s.val:3} {s.archive or s.upstream_split}")
        log(f"TOTAL train={sum(s.train for s in SOURCES)} internal_val={sum(s.val for s in SOURCES)}")
        log("Excluded: ALL WildFake DALLE / COCO archives; SID label 2. No downloads started.")
        return
    project = args.project.resolve()
    root = (project / args.data_dir).resolve()
    root.relative_to(project)  # Keep writable generated data within the project.
    test_root = (project / args.test_root).resolve()
    if root == project or root == test_root or root in test_root.parents or test_root in root.parents:
        parser.error("--data-dir must be a separate subdirectory, not the project or test directory")
    base_config = project / args.base_config
    if not base_config.is_file():
        parser.error(f"Missing base config: {base_config}")
    root.mkdir(parents=True, exist_ok=True)
    excluded, test_count = test_fingerprints(test_root, root / "test_hashes.json")
    state = State(project, root, args.seed, excluded)
    try:
        if not args.existing_only:
            prepare_cifake(state, args.cifake_root)
            prepare_sid(state)
            for source in SOURCES:
                if source.archive:
                    prepare_wildfake(state, source, project / args.archive_dir, args.download_mode, args.workers)
        write_outputs(state, base_config, test_root, test_count)
    finally:
        state.close()


if __name__ == "__main__":
    main()

"""Download pinned sources and official public assets; verify before extraction.

Archives, model weights and dataset remain local and are not redistributed.
First use needs approximately 12 GB disk including unpacked smoke data.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import tarfile
import time
import urllib.request
import zipfile

COMMIT = "bec29660f5ea46ac73db8e2d0c33c8d1a72c23ad"
MODEL_REV = "15c0770ebd233665214590bb2190a90907499f9e"
DATA_REV = "a76f840b65e972bc45e56c2adced897498e9a026"
BASE = f"https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/{DATA_REV}/openscene-v1.1"
ASSETS = {
    "model_state_dict.pt": (f"https://huggingface.co/AFARI-Research/WA-JEPA/resolve/{MODEL_REV}/model_state_dict.pt", "e1d3be013de7dbaa5a6c30925a0d2f91e6ce1a6b17a75ca73fe01a1abe6d6408"),
    "metadata_test.tgz": (f"{BASE}/openscene_metadata_test.tgz", "871ba786da584e03f5151924674ab006e83782f796a979ee5dd144b5c6dca1ab"),
    "camera_test_30.tgz": (f"{BASE}/openscene_sensor_test_camera/openscene_sensor_test_camera_30.tgz", "67a9c6ca0cb7ed650e93e2790425f8676f4749125e72f6f9f9f003eedfc36c57"),
    "metadata_mini.tgz": (f"{BASE}/openscene_metadata_mini.tgz", "c2e08867e062ba916eccd56df191e728d42fb94de854c28f2226eed14cecd8be"),
    "camera_mini_5.tgz": (f"{BASE}/openscene_sensor_mini_camera/openscene_sensor_mini_camera_5.tgz", "74df953723507a8dcd778c6d1feb2ef2a3e57b799a6e3b35f1728cb72384a595"),
}


def download(url, destination, expected=None, expected_size=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if expected_size and not destination.exists():
        # Bounded transfers tolerate proxies that terminate very large streams.
        partial = destination.with_suffix(destination.suffix + ".partial")
        failures = 0
        last_print = time.monotonic()
        while (partial.stat().st_size if partial.exists() else 0) < expected_size:
            offset = partial.stat().st_size if partial.exists() else 0
            end = min(offset + 64*1024*1024, expected_size) - 1
            separator = "&" if "?" in url else "?"
            request = urllib.request.Request(f"{url}{separator}download=true&range={offset}-{end}",
                                             headers={"Range":f"bytes={offset}-{end}"})
            try:
                with urllib.request.urlopen(request,timeout=60) as response:
                    content_range = response.headers.get("Content-Range", "")
                    if response.status!=206 or not content_range.startswith(f"bytes {offset}-"):
                        raise IOError("Server did not honor requested resume offset")
                    remaining = end-offset+1
                    with partial.open("ab" if offset else "wb") as stream:
                        while remaining:
                            block=response.read(min(4*1024*1024,remaining))
                            if not block:
                                raise IOError("Truncated range response")
                            stream.write(block);remaining-=len(block)
                failures=0
                if time.monotonic()-last_print>30:
                    print(f"{destination.name}: {partial.stat().st_size/1e6:.0f}/{expected_size/1e6:.0f} MB",flush=True)
                    last_print=time.monotonic()
            except Exception as error:
                failures+=1
                print(f"range retry {destination.name}: {type(error).__name__}",flush=True)
                if failures>=5:
                    raise
        partial.replace(destination)
    if not destination.exists():
        partial = destination.with_suffix(destination.suffix + ".partial")
        for attempt in range(5):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                # Unique query prevents expired/cached signed URLs and stale Range replies.
                separator = "&" if "?" in url else "?"
                request_url = f"{url}{separator}download=true&resume={offset}&attempt={attempt}"
                request = urllib.request.Request(request_url, headers={"Range": f"bytes={offset}-"} if offset else {})
                with urllib.request.urlopen(request, timeout=90) as response:
                    append = offset > 0 and response.status == 206
                    if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise IOError("Server returned the wrong resume offset")
                    offset = offset if append else 0
                    total = offset + int(response.headers.get("Content-Length", 0))
                    done = offset
                    last_print = time.monotonic()
                    with partial.open("ab" if append else "wb") as stream:
                        while block := response.read(4 * 1024 * 1024):
                            stream.write(block)
                            done += len(block)
                            if time.monotonic() - last_print > 30:
                                print(f"{destination.name}: {done/1e6:.0f}/{total/1e6:.0f} MB", flush=True)
                                last_print = time.monotonic()
                    if total and done != total:
                        raise IOError("Incomplete download")
                partial.replace(destination)
                break
            except Exception as error:
                print(f"retry {destination.name}: {type(error).__name__}: {error}", flush=True)
                if attempt == 4:
                    raise
    with destination.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    if expected and checksum != expected:
        raise ValueError(f"SHA256 mismatch: {destination}")
    print(f"verified {destination.name}", flush=True)
    return {"file": destination.name, "url": url, "sha256": checksum, "bytes": destination.stat().st_size}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--sources-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    third = Path("third_party").resolve()
    third.mkdir(exist_ok=True)
    url = f"https://codeload.github.com/AFARI-Research/WA-JEPA/zip/{COMMIT}"
    record = download(url, third/"wajepa.zip", "d007420c1d76d8993919428c8aa786507e32ed4f5179242b73334b88a1da21ca")
    upstream = third/f"WA-JEPA-{COMMIT}"
    if not upstream.exists():
        with zipfile.ZipFile(third/"wajepa.zip") as archive:
            archive.extractall(third)
    filters = third/"navsim-v2.2-filters"
    for name in ("navtrain", "navtest"):
        url = f"https://raw.githubusercontent.com/autonomousvision/navsim/v2.2/navsim/planning/script/config/common/train_test_split/scene_filter/{name}.yaml"
        download(url, filters/(name+".yaml"))
    (third/"source_provenance.json").write_text(json.dumps(record, indent=2))
    if not args.sources_only:
        def fetch(item):
            name, (url, checksum) = item
            return download(url, root/"archives"/name, checksum)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            records = list(executor.map(fetch, ASSETS.items()))
        (root/"provenance.json").write_text(json.dumps(records, indent=2))
        for name in ASSETS:
            if name.endswith(".tgz"):
                marker = root/(name+".extracted")
                if not marker.exists():
                    with tarfile.open(root/"archives"/name) as archive:
                        archive.extractall(root/"openscene", filter="data")
                    marker.touch()
    for split in ("train", "test"):
        if args.sources_only and Path(f"local_{split}.json").exists():
            continue
        sensor_split = "mini" if split == "train" else "test"
        config = {"upstream": str(upstream), "checkpoint": str(root/"archives/model_state_dict.pt"),
                  "metadata": str(root/f"openscene/openscene-v1.1/meta_datas/{sensor_split}"),
                  "sensors": str(root/f"openscene/openscene-v1.1/sensor_blobs/{sensor_split}"),
                  "scene_filter": str(filters/("navtrain.yaml" if split=="train" else "navtest.yaml")),
                  "stride": 5 if split=="train" else 3}
        Path(f"local_{split}.json").write_text(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()

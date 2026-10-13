"""Expand the local subset by named official shards, pinned HF SHA256 verified.

Example: python scripts/expand_data.py --root data --mini 6 4 21 19 --test 19 25 15
Archive selection is based on download size, not observed test model errors.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import tarfile
import urllib.request
from bootstrap import download, DATA_REV


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--mini", type=int, nargs="*", default=[])
    parser.add_argument("--test", type=int, nargs="*", default=[])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--backend", choices=["http","hf"], default="http")
    args = parser.parse_args()
    if args.backend=="hf":
        # Windows registry proxy settings are not automatically read by httpx.
        proxy=urllib.request.getproxies().get("https")
        if proxy:
            os.environ.setdefault("HTTPS_PROXY",proxy)
            os.environ.setdefault("HTTP_PROXY",proxy)
        os.environ.setdefault("HF_HUB_DISABLE_XET","1")
        from huggingface_hub import hf_hub_download
    requested = []
    for split in ("mini", "test"):
        ids = getattr(args, split)
        if not ids:
            continue
        url = f"https://huggingface.co/api/datasets/OpenDriveLab/OpenScene/tree/{DATA_REV}/openscene-v1.1/openscene_sensor_{split}_camera?limit=100"
        with urllib.request.urlopen(url, timeout=60) as response:
            tree = {x["path"].split("/")[-1]: x for x in json.load(response) if x["type"]=="file"}
        for index in ids:
            name = f"openscene_sensor_{split}_camera_{index}.tgz"
            info = tree[name]
            requested.append((info["path"], info["lfs"]["oid"], info["size"]))

    def fetch(item):
        path, checksum, size = item
        url = f"https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/{DATA_REV}/{path}"
        filename = args.root/"archives"/Path(path).name
        if args.backend=="hf":
            cached=Path(hf_hub_download("OpenDriveLab/OpenScene",path,repo_type="dataset",revision=DATA_REV,
                                       cache_dir=args.root/"hf-cache"))
            record=download(url,cached,checksum)
            record["file"]=Path(path).name
            filename=cached
        else:
            record = download(url,filename,checksum,expected_size=size)
        marker = args.root/(Path(path).name+".extracted")
        if not marker.exists():
            with tarfile.open(filename) as archive:
                archive.extractall(args.root/"openscene",filter="data")
            marker.touch()
        print(f"extracted {filename.name}",flush=True)
        return record

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        records = list(executor.map(fetch,requested))
    (args.root/"expanded_provenance.json").write_text(json.dumps(records,indent=2))


if __name__=="__main__":
    main()

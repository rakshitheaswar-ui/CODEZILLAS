#!/usr/bin/env python3
"""
download_data.py

Usage examples:
  python download_data.py --provider drive --url "https://drive.google.com/..." --out data/merged.pkl
  python download_data.py --provider dropbox --url "https://www.dropbox.com/s/....?dl=0" --out data/merged.pkl
  python download_data.py --provider http --url "https://github.com/.../releases/download/v1/asset.zip" --out data/asset.zip
  python download_data.py --provider kaggle --kaggle-dataset "username/dataset-name" --outdir data
"""
import os
import argparse
import hashlib
import requests
import shutil
import subprocess

def sha256(filepath, block_size=65536):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()

def download_stream(url, out_path, chunk_size=32768):
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)

def download_google_drive(share_url, out_path):
    try:
        import gdown
    except Exception:
        raise RuntimeError("gdown is required for Google Drive downloads. Install with `pip install gdown`.")
    # gdown accepts share links directly and handles large-file confirmation.
    gdown.download(share_url, out_path, quiet=False, fuzzy=True)

def download_dropbox(share_url, out_path):
    # Make sure dl=1 to force download
    url = share_url.replace("?dl=0", "?dl=1")
    download_stream(url, out_path)

def download_http(url, out_path):
    download_stream(url, out_path)

def download_kaggle(dataset, outdir):
    # requires kaggle CLI configured (KAGGLE_USERNAME & KAGGLE_KEY in ~/.kaggle/kaggle.json)
    os.makedirs(outdir, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", dataset, "-p", outdir, "--unzip"]
    subprocess.check_call(cmd)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--provider", choices=["drive","dropbox","http","kaggle"], required=True)
    p.add_argument("--url", help="Shareable link or direct url (for drive/dropbox/http).")
    p.add_argument("--out", help="Destination filepath or folder (for kaggle use outdir).", required=True)
    p.add_argument("--kaggle-dataset", help="For provider kaggle: owner/dataset-id")
    p.add_argument("--sha256", help="Optional: expected sha256 to verify file")
    args = p.parse_args()

    if args.provider == "drive":
        print("Downloading from Google Drive...")
        download_google_drive(args.url, args.out)
    elif args.provider == "dropbox":
        print("Downloading from Dropbox...")
        download_dropbox(args.url, args.out)
    elif args.provider == "http":
        print("Downloading from HTTP url...")
        download_http(args.url, args.out)
    elif args.provider == "kaggle":
        if not args.kaggle_dataset:
            raise ValueError("For kaggle provider you must pass --kaggle-dataset owner/dataset")
        print("Downloading from Kaggle...")
        download_kaggle(args.kaggle_dataset, args.out)
    else:
        raise ValueError("Unknown provider")

    if args.sha256:
        got = sha256(args.out)
        print("Expected sha256:", args.sha256)
        print("Downloaded sha256:", got)
        if got != args.sha256:
            raise RuntimeError("SHA256 mismatch! Remove the bad file and try again.")
    print("Done. File saved to:", args.out)

if __name__ == "__main__":
    main()

"""Explicitly prepare the pinned official 4B test model; never run on import."""

import argparse
import hashlib
import json
from pathlib import Path

import httpx

REPOSITORY = "Qwen/Qwen3-4B-GGUF"
REVISION = "bc640142c66e1fdd12af0bd68f40445458f3869b"
FILENAME = "Qwen3-4B-Q4_K_M.gguf"
SHA256 = "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5"
SIZE = 2497280256


def prepare(cache: Path, output: Path, url="http://127.0.0.1:11434", model="qwen3-hf:4b"):
    from huggingface_hub import hf_hub_download

    if output.exists():
        raise ValueError("Refusing to overwrite existing model verification report")
    path = Path(hf_hub_download(REPOSITORY, FILENAME, revision=REVISION, cache_dir=cache))
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if path.stat().st_size != SIZE or digest != SHA256:
        raise ValueError("Official model size/checksum verification failed; model not imported")
    # Upload a verified file, not arbitrary filenames or remote Modelfile commands.
    with httpx.Client(trust_env=False, timeout=900) as client:
        blob_url = url.rstrip("/") + "/api/blobs/sha256:" + SHA256
        if client.head(blob_url).status_code != 200:

            def blocks():
                with path.open("rb") as source:
                    yield from iter(lambda: source.read(8 * 1024 * 1024), b"")

            response = client.post(
                blob_url, content=blocks(), headers={"Content-Length": str(SIZE)}
            )
            response.raise_for_status()
        response = client.post(
            url.rstrip("/") + "/api/create",
            json={
                "model": model,
                "files": {FILENAME: "sha256:" + SHA256},
                "stream": False,
                "parameters": {"num_ctx": 8192, "num_thread": 4},
            },
        )
        response.raise_for_status()
        tags = client.get(url.rstrip("/") + "/api/tags").json()["models"]
        installed = next(m for m in tags if m["name"] == model)
        version = client.get(url.rstrip("/") + "/api/version").json()
    result = dict(
        repository=REPOSITORY,
        revision=REVISION,
        filename=FILENAME,
        sha256=SHA256,
        bytes=SIZE,
        installed_model=installed,
        ollama=version,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    print(f"Verified and imported {model}; publisher SHA-256 {SHA256}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:11434")
    args = parser.parse_args()
    prepare(args.cache, args.output, args.url)


if __name__ == "__main__":
    main()

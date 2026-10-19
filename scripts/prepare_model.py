"""Explicitly prepare the pinned 4B test model; never run on import."""

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


INSTRUCT_PROFILE = {
    "repository": "unsloth/Qwen3-4B-Instruct-2507-GGUF",
    "revision": "a06e946bb6b655725eafa393f4a9745d460374c9",
    "filename": "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    "sha256": "3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597",
    "bytes": 2497281120,
    "model": "qwen3-instruct-hf:4b",
    "publisher": "Unsloth community quantization of Qwen3-4B-Instruct-2507",
}


def prepare(cache: Path, output: Path, url="http://127.0.0.1:11434", model=None, *, profile="base"):
    from huggingface_hub import hf_hub_download

    if profile not in {"base", "instruct"}:
        raise ValueError("Unknown model profile")
    selected = (
        INSTRUCT_PROFILE
        if profile == "instruct"
        else {
            "repository": REPOSITORY,
            "revision": REVISION,
            "filename": FILENAME,
            "sha256": SHA256,
            "bytes": SIZE,
            "model": "qwen3-hf:4b",
            "publisher": "Qwen official GGUF",
        }
    )
    repository, revision, filename = (selected[k] for k in ("repository", "revision", "filename"))
    expected_digest, size = selected["sha256"], selected["bytes"]
    model = model or selected["model"]
    if output.exists():
        raise ValueError("Refusing to overwrite existing model verification report")
    path = Path(hf_hub_download(repository, filename, revision=revision, cache_dir=cache))
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if path.stat().st_size != size or digest != expected_digest:
        raise ValueError("Pinned model size/checksum verification failed; model not imported")
    # Upload a verified file, not arbitrary filenames or remote Modelfile commands.
    with httpx.Client(trust_env=False, timeout=900) as client:
        blob_url = url.rstrip("/") + "/api/blobs/sha256:" + expected_digest
        if client.head(blob_url).status_code != 200:

            def blocks():
                with path.open("rb") as source:
                    yield from iter(lambda: source.read(8 * 1024 * 1024), b"")

            response = client.post(
                blob_url, content=blocks(), headers={"Content-Length": str(size)}
            )
            response.raise_for_status()
        response = client.post(
            url.rstrip("/") + "/api/create",
            json={
                "model": model,
                "files": {filename: "sha256:" + expected_digest},
                "stream": False,
                "parameters": {"num_ctx": 8192, "num_thread": 4},
            },
        )
        response.raise_for_status()
        tags = client.get(url.rstrip("/") + "/api/tags").json()["models"]
        installed = next(m for m in tags if m["name"] == model)
        version = client.get(url.rstrip("/") + "/api/version").json()
    result = dict(
        repository=repository,
        revision=revision,
        filename=filename,
        sha256=expected_digest,
        bytes=size,
        publisher=selected["publisher"],
        profile=profile,
        installed_model=installed,
        ollama=version,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    print(f"Verified and imported {model}; publisher SHA-256 {expected_digest}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:11434")
    parser.add_argument("--profile", choices=["base", "instruct"], default="base")
    args = parser.parse_args()
    prepare(args.cache, args.output, args.url, profile=args.profile)


if __name__ == "__main__":
    main()

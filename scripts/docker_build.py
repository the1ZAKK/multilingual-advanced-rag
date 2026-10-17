"""Build locally, optionally using a trusted corporate CA and proxy DNS mapping."""

import argparse
import os
import socket
import subprocess
from pathlib import Path
from urllib.parse import urlparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="multilingual-rag:local")
    parser.add_argument("--ca-bundle", type=Path)
    parser.add_argument("--resolve-proxy-host", action="store_true")
    args = parser.parse_args()
    command = [
        "docker",
        "build",
        "--build-arg",
        "HTTPS_PROXY",
        "--build-arg",
        "HTTP_PROXY",
        "--build-arg",
        "NO_PROXY",
    ]
    if args.ca_bundle:
        if not args.ca_bundle.is_file():
            parser.error("Trusted CA bundle does not exist")
        command += ["--secret", f"id=corporate_ca,src={args.ca_bundle.resolve()}"]
    if args.resolve_proxy_host:
        host = urlparse(os.environ.get("HTTPS_PROXY", os.environ.get("https_proxy", ""))).hostname
        if host:
            command += ["--add-host", f"{host}:{socket.gethostbyname(host)}"]
    command += ["--tag", args.tag, "."]
    result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=False)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

"""Download the raw documents into ``data/raw``.

Usage: python -m manual_assistant.download
"""

import argparse
import json
import logging
import time
from pathlib import Path

import httpx

from manual_assistant.config import get_settings
from manual_assistant.sources import CALCULIX_PDF_CANDIDATES, FREECAD_API, FREECAD_PAGES

USER_AGENT = "engineering-manual-assistant/0.1 (portfolio RAG demo; polite single-threaded fetch)"
log = logging.getLogger(__name__)


def download_freecad(client: httpx.Client, out_dir: Path, delay: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for title in FREECAD_PAGES:
        target = out_dir / f"{title}.json"
        if target.exists():
            continue
        params = {
            "action": "parse",
            "page": title,
            "prop": "text|sections|displaytitle",
            "redirects": 1,
            "format": "json",
            "formatversion": 2,
        }
        response = client.get(FREECAD_API, params=params)
        response.raise_for_status()
        payload = response.json()
        if "error" in payload:
            log.warning("skipping %s: %s", title, payload["error"].get("info"))
        else:
            target.write_text(json.dumps(payload["parse"]), encoding="utf-8")
            log.info("saved %s", title)
        time.sleep(delay)


def download_calculix(client: httpx.Client, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "ccx.pdf"
    if target.exists():
        return
    for url in CALCULIX_PDF_CANDIDATES:
        response = client.get(url)
        if response.status_code == 200 and response.content.startswith(b"%PDF"):
            target.write_bytes(response.content)
            (out_dir / "source.json").write_text(json.dumps({"url": url}), encoding="utf-8")
            log.info("saved %s (%d bytes)", url, len(response.content))
            return
        log.info("not available: %s (%s)", url, response.status_code)
    raise RuntimeError("no CalculiX manual PDF could be downloaded")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between wiki requests")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    raw_dir = get_settings().data_dir / "raw"
    headers = {"User-Agent": USER_AGENT}
    with httpx.Client(headers=headers, timeout=60, follow_redirects=True) as client:
        download_freecad(client, raw_dir / "freecad", args.delay)
        download_calculix(client, raw_dir / "calculix")


if __name__ == "__main__":
    main()

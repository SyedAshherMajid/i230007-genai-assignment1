"""Download the authors' public FS2K Google Drive archive (without credentials)."""
from __future__ import annotations

from html.parser import HTMLParser
import http.cookiejar
from pathlib import Path
import urllib.parse
import urllib.request
import zipfile

FILE_ID = "1saIMhQ3dc5_ftkfGmBPbCluRn_zy7QQp"
ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "data/raw/fs2k/FS2K.zip"


class DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.action = None
        self.fields = {}

    def handle_starttag(self, tag, attributes):
        attributes = dict(attributes)
        if tag == "form" and attributes.get("id") == "download-form":
            self.action = attributes.get("action")
        if tag == "input" and attributes.get("type") == "hidden":
            self.fields[attributes["name"]] = attributes.get("value", "")


def main() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if TARGET.exists() and zipfile.is_zipfile(TARGET):
        print(f"Archive already present: {TARGET}", flush=True)
        return
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    page = opener.open(f"https://drive.google.com/uc?export=download&id={FILE_ID}", timeout=60)
    if page.headers.get("Content-Type", "").startswith("text/html"):
        form = DownloadForm()
        form.feed(page.read().decode("utf-8", "replace"))
        if not form.action or form.fields.get("id") != FILE_ID:
            raise RuntimeError("Could not confirm the authors' public download")
        url = form.action + "?" + urllib.parse.urlencode(form.fields)
        page = opener.open(url, timeout=60)
    if page.headers.get("Content-Type", "").startswith("text/html"):
        raise RuntimeError("Google Drive returned HTML instead of the FS2K archive")
    total = int(page.headers.get("Content-Length", 0))
    print(f"Downloading FS2K archive; advertised bytes: {total}", flush=True)
    part = TARGET.with_suffix(".zip.part")
    received = 0
    with part.open("wb") as output:
        while block := page.read(4 * 1024 * 1024):
            output.write(block)
            received += len(block)
            if received // (100 * 1024 * 1024) != (received - len(block)) // (100 * 1024 * 1024):
                print(f"Downloaded {received // (1024 * 1024)} MiB", flush=True)
    if total and received != total:
        raise IOError(f"Incomplete FS2K download: {received}/{total}")
    if not zipfile.is_zipfile(part):
        raise ValueError("Downloaded FS2K file is not a ZIP archive")
    part.replace(TARGET)
    print(f"FS2K archive saved: {TARGET}", flush=True)


if __name__ == "__main__":
    main()

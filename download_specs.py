"""
Download 3GPP specifications from the official 3GPP FTP server.
Downloads ZIP files, extracts the DOCX inside, and places them in data/raw/.

We pick stable Release 18 versions (suffix 'i' series = Rel-18).
"""
import os
import io
import zipfile
import urllib.request

RAW_DIR = os.path.join("data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# 3GPP version naming:
# The version in the filename uses hex. E.g.:
# 23501-i80.zip = TS 23.501, version 18.8.0 (Rel-18)
# We pick a recent stable version per spec.
SPECS = {
    "23501-i80.zip": "https://www.3gpp.org/ftp/Specs/archive/23_series/23.501/23501-i80.zip",
    "24501-i80.zip": "https://www.3gpp.org/ftp/Specs/archive/24_series/24.501/24501-i80.zip",
    "38331-i40.zip": "https://www.3gpp.org/ftp/Specs/archive/38_series/38.331/38331-i40.zip",
    "23503-i70.zip": "https://www.3gpp.org/ftp/Specs/archive/23_series/23.503/23503-i70.zip",
}

# Fallback URLs if the above version doesn't exist
FALLBACK_SPECS = {
    "23501-i50.zip": "https://www.3gpp.org/ftp/Specs/archive/23_series/23.501/23501-i50.zip",
    "24501-i50.zip": "https://www.3gpp.org/ftp/Specs/archive/24_series/24.501/24501-i50.zip",
    "38331-i20.zip": "https://www.3gpp.org/ftp/Specs/archive/38_series/38.331/38331-i20.zip",
    "23503-i50.zip": "https://www.3gpp.org/ftp/Specs/archive/23_series/23.503/23503-i50.zip",
}


def download_and_extract(name, url):
    """Download a ZIP and extract any .docx or .doc files from it."""
    print(f"Downloading {name} from {url} ...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
    except Exception as e:
        print(f"  FAILED: {e}")
        return False

    print(f"  Downloaded {len(data) / 1024:.1f} KB. Extracting...")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            extracted = False
            for member in zf.namelist():
                lower = member.lower()
                if lower.endswith((".docx", ".doc")):
                    # Extract to data/raw/ with a clean name
                    out_name = os.path.basename(member)
                    out_path = os.path.join(RAW_DIR, out_name)
                    with zf.open(member) as src, open(out_path, "wb") as dst:
                        dst.write(src.read())
                    print(f"  Extracted: {out_name}")
                    extracted = True
            if not extracted:
                # Maybe it's a PDF inside? Or the doc is at the root
                for member in zf.namelist():
                    lower = member.lower()
                    if lower.endswith(".pdf") or (not lower.endswith(".zip") and "/" not in member):
                        out_name = os.path.basename(member)
                        out_path = os.path.join(RAW_DIR, out_name)
                        with zf.open(member) as src, open(out_path, "wb") as dst:
                            dst.write(src.read())
                        print(f"  Extracted: {out_name}")
                        extracted = True
            if not extracted:
                print(f"  WARNING: No extractable doc/docx/pdf found in {name}")
                print(f"  Contents: {zf.namelist()[:10]}")
    except zipfile.BadZipFile:
        # Maybe it's a direct file download, not a ZIP
        ext = ".docx" if b"PK" not in data[:4] else ".bin"
        out_path = os.path.join(RAW_DIR, name.replace(".zip", ext))
        with open(out_path, "wb") as f:
            f.write(data)
        print(f"  Saved as raw file: {out_path}")
    return True


def main():
    print("=" * 60)
    print("3GPP Specification Downloader")
    print("=" * 60)
    
    success_count = 0
    for name, url in SPECS.items():
        ok = download_and_extract(name, url)
        if not ok:
            # Try fallback
            spec_prefix = name[:5]  # e.g., "23501"
            fallback = {k: v for k, v in FALLBACK_SPECS.items() if k.startswith(spec_prefix)}
            for fb_name, fb_url in fallback.items():
                print(f"  Trying fallback: {fb_name}")
                ok = download_and_extract(fb_name, fb_url)
                if ok:
                    break
        if ok:
            success_count += 1

    print(f"\n{'=' * 60}")
    print(f"Downloaded {success_count}/{len(SPECS)} specs to {RAW_DIR}/")
    
    # List what we got
    files = os.listdir(RAW_DIR)
    if files:
        print(f"Files ready for ingestion:")
        for f in sorted(files):
            size = os.path.getsize(os.path.join(RAW_DIR, f))
            print(f"  {f} ({size / 1024:.1f} KB)")
    else:
        print("No files downloaded. Check your internet connection.")


if __name__ == "__main__":
    main()

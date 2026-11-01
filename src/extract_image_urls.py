# ============================================================
# extract_image_urls.py
# ============================================================
# Purpose: one-time utility, not part of the main pipeline
# Reads: meta_Health_and_Personal_Care.jsonl
# Produces: src/io/image_urls.csv  (columns: asin, image_url)

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

INPUT_PATH = HERE / "io" / "meta_Health_and_Personal_Care.jsonl"
OUTPUT_PATH = HERE / "io" / "image_urls.csv"

total = 0
written = 0
skipped = 0

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(INPUT_PATH, "r", encoding="utf-8") as infile, \
     open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as outfile:

    writer = csv.writer(outfile)
    writer.writerow(["asin", "image_url"])

    for lineno, raw in enumerate(infile, start=1):
        total += 1

        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            print(f"Warning: skipping line {lineno} — JSON decode error: {exc}", file=sys.stderr)
            skipped += 1
            continue

        asin = record.get("parent_asin")
        if not asin:
            skipped += 1
            continue

        images = record.get("images")
        if not isinstance(images, list) or len(images) == 0:
            skipped += 1
            continue

        first = images[0]
        if not isinstance(first, dict):
            skipped += 1
            continue

        url = ""
        for key in ("hi_res", "large", "thumb"):
            val = first.get(key)
            if val and str(val).startswith("http"):
                url = val
                break

        if not url:
            skipped += 1
            continue

        writer.writerow([asin, url])
        written += 1

print(f"Done. Lines read: {total} | Rows written: {written} | Rows skipped: {skipped}")

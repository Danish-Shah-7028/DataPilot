"""Build the evaluation subset from the Spider dev set.

The subset is a fixed-seed random sample of the 1,034 dev questions, so anyone can
rebuild exactly the same 200 questions. Only the SQLite databases those questions use
are extracted.
"""

import argparse
import json
import random
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import httpx

# Official download linked from https://yale-lily.github.io/spider (CC BY-SA 4.0).
SPIDER_URL = (
    "https://drive.usercontent.google.com/download"
    "?id=1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J&export=download&confirm=t"
)
SPIDER_ZIP_BYTES = 205_800_266


def download_spider(raw_dir: Path) -> Path:
    target = raw_dir / "spider_data.zip"
    if target.exists() and target.stat().st_size == SPIDER_ZIP_BYTES:
        return target
    raw_dir.mkdir(parents=True, exist_ok=True)
    print("downloading Spider (about 200 MB)")
    with httpx.stream("GET", SPIDER_URL, follow_redirects=True, timeout=300) as response:
        response.raise_for_status()
        with target.open("wb") as handle:
            for chunk in response.iter_bytes():
                handle.write(chunk)
    return target


def read_dev_questions(archive: zipfile.ZipFile) -> list[dict]:
    name = next(item for item in archive.namelist() if item.endswith("/dev.json"))
    return json.loads(archive.read(name))


def select_subset(questions: list[dict], size: int, seed: int) -> list[dict]:
    chosen = sorted(random.Random(seed).sample(range(len(questions)), size))
    return [
        {
            "id": f"spider-dev-{index:04d}",
            "db_id": questions[index]["db_id"],
            "question": questions[index]["question"],
            "gold_sql": questions[index]["query"],
        }
        for index in chosen
    ]


def extract_databases(archive: zipfile.ZipFile, db_ids: set[str], out_dir: Path) -> None:
    for name in archive.namelist():
        # Looking for spider_data/database/<db_id>/<db_id>.sqlite
        parts = name.split("/")
        if len(parts) < 4 or parts[-3] != "database" or parts[-2] not in db_ids:
            continue
        if parts[-1] == f"{parts[-2]}.sqlite":
            target = out_dir / parts[-2] / parts[-1]
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as source, target.open("wb") as handle:
                shutil.copyfileobj(source, handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the Spider dev subset for evaluation.")
    parser.add_argument("--size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/benchmark"))
    parser.add_argument("--spider-zip", type=Path, help="use an already downloaded spider_data.zip")
    args = parser.parse_args()

    zip_path = args.spider_zip or download_spider(args.raw_dir)
    with zipfile.ZipFile(zip_path) as archive:
        questions = read_dev_questions(archive)
        subset = select_subset(questions, args.size, args.seed)
        db_ids = {item["db_id"] for item in subset}
        extract_databases(archive, db_ids, args.out_dir / "databases")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    subset_path = args.out_dir / "spider_dev_subset.jsonl"
    with subset_path.open("w", encoding="utf-8") as handle:
        for item in subset:
            handle.write(json.dumps(item) + "\n")

    per_db = Counter(item["db_id"] for item in subset)
    print(f"Selected {len(subset)} of {len(questions)} dev questions (seed {args.seed}).")
    print(f"They use {len(db_ids)} databases; most common: {per_db.most_common(5)}")
    print(f"Wrote {subset_path}")

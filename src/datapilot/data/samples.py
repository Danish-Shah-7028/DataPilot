"""Download the public datasets behind the demo and build the small samples in data/samples.

Sources (both CC BY 4.0, see data/samples/README.md):
- UCI Online Retail II: transactions of a UK online gift retailer, Dec 2009 - Dec 2011.
- UCI Iranian Churn: 3,150 telecom customers with a churn label.
"""

import argparse
import io
import sqlite3
import zipfile
from contextlib import closing
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from datapilot.data.loader import safe_identifier

ONLINE_RETAIL_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
CHURN_URL = "https://archive.ics.uci.edu/static/public/563/iranian+churn+dataset.zip"

RETAIL_COLUMNS = {
    "Invoice": "invoice_no",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "unit_price",
    "Customer ID": "customer_id",
    "Country": "country",
}

# (region, continent) for every country that appears in Online Retail II.
COUNTRY_REGIONS = {
    "United Kingdom": ("United Kingdom", "Europe"),
    "EIRE": ("Western Europe", "Europe"),
    "Channel Islands": ("Western Europe", "Europe"),
    "France": ("Western Europe", "Europe"),
    "Belgium": ("Western Europe", "Europe"),
    "Netherlands": ("Western Europe", "Europe"),
    "Germany": ("Western Europe", "Europe"),
    "Switzerland": ("Western Europe", "Europe"),
    "Austria": ("Western Europe", "Europe"),
    "Denmark": ("Northern Europe", "Europe"),
    "Norway": ("Northern Europe", "Europe"),
    "Sweden": ("Northern Europe", "Europe"),
    "Finland": ("Northern Europe", "Europe"),
    "Iceland": ("Northern Europe", "Europe"),
    "Lithuania": ("Northern Europe", "Europe"),
    "Spain": ("Southern Europe", "Europe"),
    "Portugal": ("Southern Europe", "Europe"),
    "Italy": ("Southern Europe", "Europe"),
    "Greece": ("Southern Europe", "Europe"),
    "Malta": ("Southern Europe", "Europe"),
    "Cyprus": ("Southern Europe", "Europe"),
    "Poland": ("Eastern Europe", "Europe"),
    "Czech Republic": ("Eastern Europe", "Europe"),
    "European Community": ("Western Europe", "Europe"),
    "Israel": ("Middle East", "Asia"),
    "Lebanon": ("Middle East", "Asia"),
    "Bahrain": ("Middle East", "Asia"),
    "Saudi Arabia": ("Middle East", "Asia"),
    "United Arab Emirates": ("Middle East", "Asia"),
    "Japan": ("Asia Pacific", "Asia"),
    "Korea": ("Asia Pacific", "Asia"),
    "Hong Kong": ("Asia Pacific", "Asia"),
    "Singapore": ("Asia Pacific", "Asia"),
    "Thailand": ("Asia Pacific", "Asia"),
    "Australia": ("Asia Pacific", "Oceania"),
    "USA": ("Americas", "North America"),
    "Canada": ("Americas", "North America"),
    "Bermuda": ("Americas", "North America"),
    "West Indies": ("Americas", "North America"),
    "Brazil": ("Americas", "South America"),
    "RSA": ("Africa", "Africa"),
    "Nigeria": ("Africa", "Africa"),
}


def download(url: str, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / url.rsplit("/", 1)[-1].replace("+", "_")
    if target.exists():
        return target
    print(f"downloading {url}")
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as response:
        response.raise_for_status()
        with target.open("wb") as handle:
            for chunk in response.iter_bytes():
                handle.write(chunk)
    return target


def read_member(archive_path: Path, suffix: str) -> bytes:
    with zipfile.ZipFile(archive_path) as archive:
        member = next(name for name in archive.namelist() if name.lower().endswith(suffix))
        return archive.read(member)


def read_online_retail(archive_path: Path) -> pd.DataFrame:
    print("reading Online Retail II workbook (about a million rows, takes a few minutes)")
    sheets = pd.read_excel(io.BytesIO(read_member(archive_path, ".xlsx")), sheet_name=None)
    frame = pd.concat(sheets.values(), ignore_index=True).rename(columns=RETAIL_COLUMNS)
    frame["invoice_no"] = frame["invoice_no"].astype(str)
    frame["stock_code"] = frame["stock_code"].astype(str)
    frame["description"] = frame["description"].astype("string").str.strip()
    frame["customer_id"] = frame["customer_id"].astype("Int64")
    frame["invoice_date"] = pd.to_datetime(frame["invoice_date"])
    return frame[list(RETAIL_COLUMNS.values())]


def keep_completed_sales(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop cancellations and bookkeeping lines (postage, fees, bad-debt adjustments).

    A random sample keeps a cancellation without the order it cancels, which produces
    nonsense like negative monthly revenue, so the demo data only holds completed sales.
    """
    stock_code = frame["stock_code"].str.upper()
    product_line = stock_code.str.contains(r"\d") & ~stock_code.str.startswith(("TEST", "ADJUST"))
    keep = (
        ~frame["invoice_no"].str.startswith("C")
        & (frame["quantity"] > 0)
        & (frame["unit_price"] > 0)
        & product_line
    )
    return frame[keep]


def sample_invoices(frame: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    """Keep a random subset of whole invoices so every kept order is complete."""
    invoices = np.sort(frame["invoice_no"].unique())
    rng = np.random.default_rng(seed)
    keep = rng.choice(invoices, size=max(1, round(len(invoices) * fraction)), replace=False)
    sampled = frame[frame["invoice_no"].isin(set(keep))]
    return sampled.sort_values(["invoice_date", "invoice_no"]).reset_index(drop=True)


def build_retail_tables(lines: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split the flat transaction file into a small normalised schema."""
    invoices = (
        lines.groupby("invoice_no", sort=False)
        .agg(
            invoice_date=("invoice_date", "first"),
            customer_id=("customer_id", "first"),
            country=("country", "first"),
        )
        .reset_index()
    )
    products = (
        lines.dropna(subset=["description"])
        .groupby("stock_code")["description"]
        .agg(lambda values: values.value_counts().index[0])
        .reset_index()
    )
    known = invoices.dropna(subset=["customer_id"])
    customers = (
        known.groupby("customer_id")
        .agg(
            country=("country", lambda values: values.value_counts().index[0]),
            first_invoice_date=("invoice_date", "min"),
        )
        .reset_index()
    )
    invoice_lines = lines[["invoice_no", "stock_code", "quantity", "unit_price"]].copy()
    invoice_lines.insert(0, "line_id", range(1, len(invoice_lines) + 1))
    regions = pd.DataFrame(
        [
            (country, *COUNTRY_REGIONS.get(country, ("Other", "Unknown")))
            for country in sorted(lines["country"].unique())
        ],
        columns=["country", "region", "continent"],
    )
    return {
        "customers": customers,
        "products": products,
        "invoices": invoices,
        "invoice_lines": invoice_lines,
        "country_regions": regions,
    }


def write_sqlite(tables: dict[str, pd.DataFrame], path: Path) -> None:
    path.unlink(missing_ok=True)
    with closing(sqlite3.connect(path)) as connection:
        for name, frame in tables.items():
            frame.to_sql(name, connection, index=False)
        connection.commit()


def read_churn(archive_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(read_member(archive_path, ".csv")))
    taken: set[str] = set()
    frame.columns = [safe_identifier(column, taken) for column in frame.columns]
    frame.insert(0, "customer_id", range(1, len(frame) + 1))
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the demo datasets in data/samples.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/samples"))
    parser.add_argument("--fraction", type=float, default=0.04, help="share of invoices kept")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    retail = keep_completed_sales(read_online_retail(download(ONLINE_RETAIL_URL, args.raw_dir)))
    lines = sample_invoices(retail, args.fraction, args.seed)
    lines.to_csv(args.out_dir / "online_retail.csv", index=False)
    tables = build_retail_tables(lines)
    write_sqlite(tables, args.out_dir / "retail.sqlite")
    print(f"online retail: kept {len(lines):,} of {len(retail):,} lines")
    for name, frame in tables.items():
        print(f"  retail.sqlite/{name}: {len(frame):,} rows")

    churn = read_churn(download(CHURN_URL, args.raw_dir))
    churn.to_excel(args.out_dir / "customer_churn.xlsx", index=False, sheet_name="customers")
    print(f"customer churn: {len(churn):,} customers, churn rate {churn['churn'].mean():.1%}")

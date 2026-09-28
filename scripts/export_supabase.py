"""Export a frozen, read-only snapshot of the Supabase price tables to Parquet.

Usage (from the ADAA folder):
    .venv/bin/python scripts/export_supabase.py                    # full catalogue
    .venv/bin/python scripts/export_supabase.py --sample-buckets 4 # 20% SKU sample (4 of 20 hash buckets)

Requires SUPABASE_DB_URL in ADAA/.env. The session is forced read-only, so the
database rejects any INSERT/UPDATE/DELETE/DDL even if one were issued.
The submitted snapshot is the 20% sample (hash buckets 0-3 of abs(hashtext(sku)) % 20).
"""

import argparse
import io
import os
import sys
from pathlib import Path

import pandas as pd
import psycopg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data"
OUT.mkdir(exist_ok=True)

START_DATE = "2026-05-01"
MIN_PRICE = 50
MIN_BUYABLE_OBS = 20
PRODUCT_TYPES = (
    "VISUAL", "COMPUTERS", "COMMUNICATIONS", "AUDIO", "GAMES HARDWARE",
    "WHITEGOODS", "SMALL APPLIANCES", "CAMERAS", "Wearables & Outdoor", "SMART HOME",
)
TYPES_SQL = ", ".join("'" + t.replace("'", "''") + "'" for t in PRODUCT_TYPES)


def build_queries(sample_buckets: int | None) -> dict[str, str]:
    sample = "" if sample_buckets is None else f"AND abs(hashtext(sku)) % 20 < {sample_buckets}"
    return {name: sql.replace("/*SAMPLE*/", sample) for name, sql in QUERIES.items()}


# SKUs in the 10 appliance categories that were buyable (>= $50) on enough scrape days.
ELIGIBLE_SKUS = f"""
    SELECT sku
    FROM public."All JB Hifi Products"
    WHERE date >= '{START_DATE}'
      AND product_type IN ({TYPES_SQL})
      AND jb_listing_cta = 'Buy'
      AND price >= {MIN_PRICE}
      /*SAMPLE*/
    GROUP BY sku
    HAVING count(DISTINCT date) >= {MIN_BUYABLE_OBS}
"""

QUERIES = {
    # One row per SKU per scrape date (duplicates collapsed to the minimum price).
    # buyable is NULL before June 2026, when the scraper did not yet record listing status.
    "jb_prices": f"""
        WITH eligible AS ({ELIGIBLE_SKUS})
        SELECT p.sku,
               p.date,
               min(p.price)                                     AS price,
               bool_or(coalesce(p.on_promotion, false))::int    AS on_promotion,
               bool_or(p.jb_listing_cta = 'Buy')::int           AS buyable
        FROM public."All JB Hifi Products" p
        JOIN eligible e USING (sku)
        WHERE p.date >= '{START_DATE}' AND p.price > 0
        GROUP BY p.sku, p.date
    """,
    # Static attributes per SKU, taken from its most recent row.
    "jb_products": f"""
        WITH eligible AS ({ELIGIBLE_SKUS})
        SELECT DISTINCT ON (p.sku)
               p.sku, p.title, p.product_type, p.vendor,
               upper(trim(p.model_number)) AS model_number
        FROM public."All JB Hifi Products" p
        JOIN eligible e USING (sku)
        WHERE p.date >= '{START_DATE}'
        ORDER BY p.sku, p.date DESC
    """,
    # Products listed on the JB website right now.
    "jb_catalog_latest": """
        SELECT sku, title, price, product_type, jb_listing_cta,
               upper(trim(model_number)) AS model_number, last_date
        FROM public.jb_catalog_latest
        WHERE true /*SAMPLE*/
    """,
    # The Good Guys prices for model numbers that JB also sells.
    "tgg_prices": f"""
        WITH jb_models AS (
            SELECT DISTINCT upper(trim(model_number)) AS model_number
            FROM public."All JB Hifi Products"
            WHERE date >= '{START_DATE}' AND product_type IN ({TYPES_SQL})
              AND model_number IS NOT NULL AND trim(model_number) <> ''
        )
        SELECT upper(trim(t.model_number)) AS model_number,
               t.date,
               min(t.price)::float8 AS tgg_price
        FROM public.the_good_guys_products t
        JOIN jb_models m ON upper(trim(t.model_number)) = m.model_number
        WHERE t.date >= '{START_DATE}' AND t.price > 0
        GROUP BY 1, 2
    """,
}


def copy_to_frame(cur: psycopg.Cursor, sql: str) -> pd.DataFrame:
    buf = io.BytesIO()
    with cur.copy(f"COPY ({sql}) TO STDOUT WITH (FORMAT csv, HEADER true)") as copy:
        for chunk in copy:
            buf.write(chunk)
    buf.seek(0)
    return pd.read_csv(buf)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-buckets", type=int, default=None,
                        help="keep SKUs with abs(hashtext(sku)) %% 20 < N (e.g. 4 = 20%% sample)")
    queries = build_queries(parser.parse_args().sample_buckets)
    load_dotenv(ROOT / ".env")
    url = os.environ.get("SUPABASE_DB_URL")
    if not url:
        sys.exit("SUPABASE_DB_URL is missing from ADAA/.env")

    with psycopg.connect(url, autocommit=False) as conn:
        conn.read_only = True
        with conn.cursor() as cur:
            cur.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
            cur.execute("SET statement_timeout = 0")
            cur.execute("SHOW transaction_read_only")
            assert cur.fetchone()[0] == "on", "session is not read-only; aborting"

            for name, sql in queries.items():
                print(f"exporting {name} ...", flush=True)
                df = copy_to_frame(cur, sql)
                path = OUT / f"{name}.parquet"
                df.to_parquet(path, index=False)
                print(f"  {len(df):,} rows -> {path.relative_to(ROOT)} "
                      f"({path.stat().st_size / 1e6:.1f} MB)", flush=True)
        conn.rollback()


if __name__ == "__main__":
    main()

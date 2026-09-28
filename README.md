# Predicting JB Hi-Fi price drops (ADAA Assignment 2, Option 2)

For a product buyable at JB Hi-Fi on day *t*, predict **when** its price will first drop by at least 5%
(1-7 days, 8-14 days, 15-28 days, or not within 28 days), **how deep** the drop will be, and whether to **BUY NOW or WAIT**.

- `notebooks/ADAA_A2_price_drop.ipynb` — the complete, self-contained notebook (open in Colab; it downloads `data/*.parquet` from this repo).
- `notebooks/ADAA_A2_price_drop.py` — the same notebook as a plain script (percent-format cells); `scripts/build_notebook.py` converts it.
- `data/` — frozen snapshot of the filtered catalogue exported read-only from my Supabase scraper database, May-Sep 2026.
- `scripts/export_supabase.py` — the read-only export (needs `SUPABASE_DB_URL` in a local `.env`; not needed to run the notebook).
- `report/` — journal, figures and `results.json`.

Data were scraped from the public JB Hi-Fi and The Good Guys websites for non-commercial academic use.

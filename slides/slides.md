# 5-minute presentation (4 slides + live notebook)

Timing: slide 1 = 1:00, slide 2 = 1:30, slide 3 = 1:30, slide 4 = 1:00. Keep the Colab notebook open beside the slides (not full screen).

---

## Slide 1 — Input, output, and how I score it (1:00)

**Input:** one JB Hi-Fi product that is buyable today (day *t*) → 23 numbers + 2 categories computed only from its past prices:
recent price changes (1-28 days), price vs its own max/min, days since last change / drop / rise, promotion flag, stock status, category, brand.

**Output:** 4 probabilities — first drop of ≥ 5% within **1-7 d / 8-14 d / 15-28 d / not in 28 d** — plus expected drop depth (%), plus **BUY NOW / WAIT**.

**Ground truth:** read off my own scrapes 28 days later. Scored with log loss, PR-AUC, calibration, and **dollars saved**.

> Speaker notes: "The input is one product on one day... the output is four probabilities that sum to one... I compare them with what my scraper saw in the following 28 days."
> Point at notebook cell 11 (labels) if asked how the truth is built.

---

## Slide 2 — Model and loss (1:30) — *the part I studied most*

* Data: my Supabase scraper, JB catalogue May-Sep 2026; 20% product sample → **3,051 products, 33,486 weekly decision points**; 83% have no drop.
* **LightGBM**: sum of trees per class → softmax → probabilities. **Loss = cross-entropy** −log p(true bucket).
  Depth: separate LightGBM with **Huber loss** (δ = 5 pp) — robust to 50% clearance drops.
* Trees because the features are tabular, mixed, non-linear; early stopping on 10% of *training products*.
* **No class weights** — I tested them: same ranking, calibration twice as bad.

> Bait for questions: "Why not a neural network?" (5 months is too little history per product; listed as future work)
> "Why is 'drop' defined with a persistence rule?" (scraper glitches create one-scrape fake drops)

---

## Slide 3 — Results, and why I trust the lower number (1:30)

| | per-category baseline | LightGBM |
|---|---|---|
| unseen products (5-fold GroupKFold) PR-AUC | 0.25 | **0.72** |
| later period (train ≤ 15 Jul, test 12-31 Aug) PR-AUC | 0.24 | **0.52** |
| log loss (GroupKFold) | 0.618 | **0.405** |
| depth error | 8.1 pp | **5.0 pp** |

Calibration: on the diagonal (ECE 0.016).
**Critical point:** GroupKFold lets other products of the *same brand in the same week* into training → brand-wide promotions leak; the time split is the honest estimate.

> Show notebook: reliability curve (cell 28), time-split table (cell 33).

---

## Slide 4 — Does it save money? + research question (1:00)

* Buy-or-wait simulation: **model saves $21.94 per purchase (CI $19-25), 80% of a perfect-foresight oracle, waiting 4.6 days on average**;
  "always wait" saves $7 after 25 days.
* **Loss ≠ objective:** cross-entropy treats a $5 and a $1,500 miss the same; worst WAIT error: a Samsung 115" TV whose promo ended ($14,921 → $26,995). Threshold τ tuned on dollars, not on loss.
* **Research question:** do The Good Guys' prices help? **No** — PR-AUC 0.80 vs 0.80; slightly better calibration only.
* Demo: `predict_drop("902842", "2026-09-28")` → Samsung 65" R95H TV, P(drop) 0.72, **WAIT**.

> End on the live demo (cell 41).

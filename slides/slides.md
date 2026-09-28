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

* Data: my Supabase scraper, JB catalogue May-Sep 2026 → **15,322 products, 175,038 weekly decision points**; 84% have no drop.
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
| unseen products (5-fold GroupKFold) PR-AUC | 0.25 | **0.85** |
| later period (train ≤ 15 Jul, test 12-31 Aug) PR-AUC | 0.24 | **0.51** |
| log loss (GroupKFold) | 0.601 | **0.271** |
| log loss (later period) | **0.580** | 0.640 |
| depth error | 7.9 pp | **3.8 pp** |

Calibration on unseen products: on the diagonal (ECE 0.008).
**Critical point:** GroupKFold lets other products of the *same brand in the same week* into training → brand-wide promotions leak; the time split is the honest estimate.
On the later period the model still ranks drops twice as well, but is over-confident (log loss worse than baseline) → needs recalibration on recent weeks.

> Show notebook: reliability curve (cell 28), time-split table (cell 33).

---

## Slide 4 — Does it save money? + research question (1:00)

* Buy-or-wait simulation: **model saves $23.57 per purchase (CI $22.31-24.82), 86% of a perfect-foresight oracle, waiting 4.1 days on average**;
  "always wait" saves $8.53 after 25 days.
* **Loss ≠ objective:** cross-entropy treats a $5 and a $1,500 miss the same; worst WAIT error: a Samsung 115" TV whose promo ended ($14,921 → $26,995). Threshold τ tuned on dollars, not on loss.
* **Research question:** do The Good Guys' prices help? **A little** — on 2,095 shared products every metric improves slightly (PR-AUC 0.900 → 0.904, +$0.80 per purchase).
* Demo: `predict_drop(892006, "2026-09-28")` → LG 97" OLED G6 at $29,995, back to full price on 25 Sep, P(drop) 0.68, **WAIT**.

> End on the live demo (cell 41).

# Q&A cheat sheet

Answer short. "I don't know, I would check X" is a valid, point-earning answer.

## Where things are in the notebook (the [n] number Colab shows next to each code cell after "Run all")

| component | code cell | one-line answer |
|---|---|---|
| Setup, constants | [1] | 5% threshold, 28-day horizon, 4 bucket names, decision dates 1 Jun - 31 Aug. |
| Data download | [2] | `load()` downloads the Parquet snapshot from GitHub if not present. |
| Partial scrape days removed | [3] | Days with < 50% of the rolling-median row count. |
| Daily grid, buyable prices | [4] | Dates × SKUs matrix; prices kept only on days the listing was buyable (unknown before June counts as buyable). |
| **Labels** | [5] | First day in t+1..t+28 with price ≤ 95% of today's, and the next scrape also ≤ 95%. Bucket 0/1/2/3; depth = 1 − drop price / price today. |
| Decision dates | [6] | First scrape of each week, 8 Jun - 31 Aug (after that the 28-day future is unknown). |
| **Features** | [7] | Rolling / cumulative windows ending at day t → no future information. |
| Dummy interface | [8] | Rule-based predictor with the same input/output, written before the model. |
| **Model, loss, optimiser** | [9] | `CLF_PARAMS` (multiclass = softmax cross-entropy), `REG_PARAMS` (huber, alpha = δ = 5), `fit_lgb` (early stopping on 10% of training SKUs). |
| Metrics | [10] | `evaluate()`: log loss, Brier, ROC-AUC, PR-AUC, ECE. |
| **Cross-validation** | [11] | 5-fold GroupKFold by SKU. |
| Skill scores, depth MAE | [12] | 1 − model / baseline. |
| Bootstrap CIs | [13] | Resample whole SKUs 200 times. |
| Calibration + per-category | [14] | Reliability curve; PR-AUC per category. |
| Feature importance | [15] | Share of split gain. |
| Timing confusion matrix | [16] | Rows = true bucket. |
| Time split + class weights + no-vendor | [17] | Train ≤ 15 Jul, test 12-31 Aug. |
| **Savings simulator** | [18]-[19] | WAIT if P(drop) > τ; τ chosen on the other 4 folds. |
| Error analysis | [20] | Costliest WAIT mistakes / missed drops. |
| Top-10 selection | [21] | One flagship per category. |
| Final model + `predict_drop` | [22]-[23] | Refit on all samples; deployment interface. |
| Top-10 timelines | [24] | Price history, real drops, out-of-fold P(drop). |
| Competitor ablation | [25] | The Good Guys price gap features, matched by model number. |

## Likely questions

**What exactly is the input?** 23 numeric + 2 categorical features for one product on one day, e.g. `ret_7` = price today / price 7 days ago − 1.

**What is the output format?** A length-4 probability vector (sums to 1) + one real number (drop depth %) + BUY/WAIT.

**What is the loss?** Multi-class cross-entropy: −log of the probability given to the true bucket. Example: true bucket gets 0.15 → loss 1.90.

**Why cross-entropy and not accuracy?** 84% of samples are "no drop", so always predicting "no drop" gets 84% accuracy and is useless. Cross-entropy rewards good probabilities.

**Why PR-AUC?** Drops are the rare class; PR-AUC focuses on how precisely the model finds them. Random guessing gives PR-AUC = drop rate (0.16).

**Why GroupKFold?** Rows of the same product in consecutive weeks are almost identical; a random split would let the model memorise the product.

**Why is the time-split score lower?** GroupKFold trains on other products from the *same weeks*; if JB discounts a whole brand that week, the model sees it. The time split never sees the test weeks → honest.

**Why is the later-period log loss worse than the baseline?** Log loss punishes confident mistakes; some brands' late-August promotions did not follow July's pattern, and the model was confidently wrong (0.615 vs 0.580). It still ranks drops far better (PR-AUC 0.51 vs 0.24). Fix: recalibrate on recent weeks (isotonic regression), retrain regularly.

**Why no class weights?** Tested on the time split: PR-AUC 0.497 vs 0.508, log loss 0.677 vs 0.615, ECE 0.078 vs 0.067 — worse on every metric.

**Why Huber for depth?** Squared error would be dominated by rare 40-60% clearance drops; Huber is quadratic below δ = 5 pp and linear above.

**How many trees / leaves?** Up to 63 leaves per tree, learning rate 0.05, ~680-850 trees per fold (early stopping, patience 50).

**How is the WAIT threshold chosen?** Maximise mean dollars saved on the other 4 folds (nested); it came out 0.15 in four folds and 0.20 in one.

**Why 0.15 and not 0.5?** Waiting when a drop does not come usually costs little (price mostly stays the same), but catching a drop saves a lot → the optimum is low.

**Why do my Colab numbers differ slightly from a local run?** Different LightGBM / scikit-learn versions; differences are in the third decimal and no conclusion changes. The journal quotes the saved Colab run.

**Is 5% arbitrary?** Yes, a design choice (smaller changes are noise / rounding); a sensitivity check with other thresholds is future work.

**Why not a neural network / LSTM?** ~5 months with scrapes every 2-3 days = ~60 points per product; too little for a sequence model to beat trees on hand-made features.

**What is censoring and how did you handle it?** For decision days after 31 Aug the 28-day future is not yet observed. I excluded them; a survival model could use them (future work).

**Did competitor prices help?** Slightly: PR-AUC 0.902 → 0.905, log loss 0.531 → 0.523, +$0.27 saved per purchase, on the 2,095 products both retailers sell. Consistent with price matching, but small next to JB's own history.

**Why did the numbers change from the first run?** The first run used a 20% product sample; the final run uses the full catalogue (15,322 products).

**Biggest limitation?** Only ~5 months: no Black Friday/Boxing Day, so no seasonality.

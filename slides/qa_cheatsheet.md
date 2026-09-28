# Q&A cheat sheet

Answer short. "I don't know, I would check X" is a valid, point-earning answer.

## Where things are in the notebook (cell numbers in the Colab notebook)

| component | cell | one-line answer |
|---|---|---|
| Data download | 5 | `load()` downloads the Parquet snapshot from GitHub if not present. |
| Partial scrape days removed | 7 | Days with < 50% of the rolling-median row count. |
| Daily grid, buyable prices | 9 | Dates × SKUs matrix; prices kept only on days the listing was buyable (unknown before June counts as buyable). |
| **Labels** | 11 | First day in t+1..t+28 with price ≤ 95% of today's, and the next scrape also ≤ 95%. Bucket 0/1/2/3; depth = 1 − drop price / price today. |
| Decision dates | 13 | First scrape of each week, 8 Jun - 31 Aug (after that the 28-day future is unknown). |
| **Features** | 15 | Rolling / cumulative windows ending at day t → no future information. |
| Dummy interface | 17 | Rule-based predictor with the same input/output, written before the model. |
| **Model, loss, optimiser** | 19 | `CLF_PARAMS` (multiclass = softmax cross-entropy), `REG_PARAMS` (huber, alpha = δ = 5), `fit_lgb` (early stopping on 10% of training SKUs). |
| Metrics | 21 | `evaluate()`: log loss, Brier, ROC-AUC, PR-AUC, ECE. |
| **Cross-validation** | 23 | 5-fold GroupKFold by SKU. |
| Bootstrap CIs | 26 | Resample whole SKUs 200 times. |
| Time split + class weights + no-vendor | 33 | Train ≤ 15 Jul, test 12-31 Aug. |
| **Savings simulator** | 35 | WAIT if P(drop) > τ; τ chosen on the other 4 folds. |
| Error analysis | 38 | Costliest WAIT mistakes / missed drops. |
| Top-10 + `predict_drop` | 40-41 | One flagship per category; final model refit on all samples. |
| Competitor ablation | 46 | The Good Guys price gap features, matched by model number. |

## Likely questions

**What exactly is the input?** 23 numeric + 2 categorical features for one product on one day, e.g. `ret_7` = price today / price 7 days ago − 1.

**What is the output format?** A length-4 probability vector (sums to 1) + one real number (drop depth %) + BUY/WAIT.

**What is the loss?** Multi-class cross-entropy: −log of the probability given to the true bucket. Example: true bucket gets 0.15 → loss 1.90.

**Why cross-entropy and not accuracy?** 83% of samples are "no drop", so always predicting "no drop" gets 83% accuracy and is useless. Cross-entropy rewards good probabilities.

**Why PR-AUC?** Drops are the rare class; PR-AUC focuses on how precisely the model finds them. Random guessing gives PR-AUC = drop rate (0.17).

**Why GroupKFold?** Rows of the same product in consecutive weeks are almost identical; a random split would let the model memorise the product.

**Why is the time-split score lower?** GroupKFold trains on other products from the *same weeks*; if JB discounts a whole brand that week, the model sees it. The time split never sees the test weeks → honest.

**Why no class weights?** Tested: same PR-AUC (0.51 vs 0.52) but ECE 0.083 vs 0.046 — probabilities get worse, and the BUY/WAIT rule relies on them.

**Why Huber for depth?** Squared error would be dominated by rare 40-60% clearance drops; Huber is quadratic below δ = 5 pp and linear above.

**How many trees / leaves?** Up to 63 leaves per tree, learning rate 0.05, ~110-170 trees per fold (early stopping, patience 50).

**How is the WAIT threshold chosen?** Maximise mean dollars saved on the other 4 folds (nested); it came out 0.20 in every fold.

**Why 0.20 and not 0.5?** Waiting when a drop does not come usually costs little (price mostly stays the same), but catching a drop saves a lot → the optimum is low.

**Is 5% arbitrary?** Yes, a design choice (smaller changes are noise / rounding); a sensitivity check with other thresholds is future work.

**Why not a neural network / LSTM?** ~5 months with scrapes every 2-3 days = ~60 points per product; too little for a sequence model to beat trees on hand-made features.

**What is censoring and how did you handle it?** For decision days after 31 Aug the 28-day future is not yet observed. I excluded them; a survival model could use them (future work).

**Did competitor prices help?** No: PR-AUC 0.804 → 0.800, log loss 0.740 → 0.735. JB's own history already contains the signal.

**Why only a 20% sample?** To keep the public snapshot small for GitHub/Colab; the export script can produce the full catalogue.

**Biggest limitation?** Only ~5 months: no Black Friday/Boxing Day, so no seasonality.

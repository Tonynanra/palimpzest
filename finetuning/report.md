## Jaccard Fuzzy matching: threshold comparison and masking of structural improvements

### 1. How extraction performance is defined

The current CUAD fine-tuning evaluation uses **fuzzy matching between model-predicted text spans and ground-truth labels**, rather than exact string matching.

The matching criterion is **word-set Jaccard similarity**, calculated as:

$$
J(P,G)=\frac{|W_P\cap W_G|}{|W_P\cup W_G|}
$$

A prediction is considered a successful match when **Jaccard similarity ≥ 0.15**, following the threshold used in DocETL (as documented in the code).

Matched ground-truth spans contribute true positives (TP); unmatched ground-truth spans contribute false negatives (FN); unmatched predictions contribute false positives (FP). Precision, recall, and F1 are then calculated from aggregate TP/FP/FN counts.

### 2. Fuzzy matching masks structural improvements

**The fuzzy-matching scheme already tolerates substantial differences between extracted text and ground-truth span boundaries.**

Crucially, matching is binary: once Jaccard similarity exceeds 0.15, the prediction receives full TP credit regardless of how closely its boundaries align with the ground truth.

For example:

| Vanilla Jaccard | SFT Jaccard | Effect on F1 |
|---|---|---|
| 0.20 | 0.95 | None |
| 0.60 | 1.00 | None |
| 0.10 | 0.90 | Improvement |
| 0.00 | 0.90 | Improvement |

*Assuming one gold span and one prediction per model.*

Therefore, **structural improvements that increase Jaccard similarity without crossing the 0.15 threshold are entirely invisible to the evaluation metric**.


### 3. A deeper look into SFT's improvements through the lens of Jaccard

Expanding vanilla model and SFT model's Jaccard scores on all predictions and labels yields the following distribution:

![img](artifacts/cuad-sft-runs/2376707/sft_plots/jaccard_distribution.png)

We can see that the SFT training is able to successfully push up many of the model's prediction toward high Jaccard than the Vanilla.

#### Appendix: SFT results with Jaccard threshold of 0.5

Previous evaluation runs used a Jaccard threshold of 0.15 that was inherited from Abacus-demo:

```python
# 0.15 is used in the Doc-ETL paper. It should be 0.5 for the actual benchmark.
IOU_THRESH = 0.15
```

As described in the comment, this `0.15` threshold is derived from Doc-ETL paper, where LLM extractions are evaluated in a **zero-shot** setting. Changing the threshold to CUAD's original `0.5` yields the following set of results:

##### Aggregate metrics

![Vanilla and SFT aggregate metrics at Jaccard threshold 0.5](artifacts/cuad-sft-runs/2376707/evaluation-jaccard-0.5/sft_plots/aggregate_metrics.png)

##### Per-category precision

![Vanilla and SFT per-category precision at Jaccard threshold 0.5](artifacts/cuad-sft-runs/2376707/evaluation-jaccard-0.5/sft_plots/per_category_precision.png)

##### Per-category recall

![Vanilla and SFT per-category recall at Jaccard threshold 0.5](artifacts/cuad-sft-runs/2376707/evaluation-jaccard-0.5/sft_plots/per_category_recall.png)

##### Per-category F1

![Vanilla and SFT per-category F1 at Jaccard threshold 0.5](artifacts/cuad-sft-runs/2376707/evaluation-jaccard-0.5/sft_plots/per_category_f1.png)

##### Per-category F1 improvement

![SFT minus vanilla per-category F1 at Jaccard threshold 0.5](artifacts/cuad-sft-runs/2376707/evaluation-jaccard-0.5/sft_plots/per_category_f1_delta.png)

##### Output format metrics

![Vanilla and SFT output format metrics for the Jaccard threshold 0.5 evaluation](artifacts/cuad-sft-runs/2376707/evaluation-jaccard-0.5/sft_plots/format_metrics.png)

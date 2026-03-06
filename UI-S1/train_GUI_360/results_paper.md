# GUI-360 Evaluation Results

## Table 1: Overall Step Success Rate (%)

| Model | Grounding | Action Pred (Visual) | Action Pred (A11y) |
|-------|:---------:|:--------------------:|:------------------:|
| GUI-360 Paper SFT | **82.30** | **50.08** | **25.78** |
| Qwen2.5-VL-7B (Base) | 42.47 | 18.05 | 14.53 |
| Qwen2.5-VL-7B + SFT v1 | 12.87 | 11.14 | 13.40 |
| Qwen2.5-VL-7B + SFT v2 (full) | 70.56 | 46.90 | 17.51 |
| Qwen2.5-VL-7B + MoE v1 LoRA | 60.32 | 33.20 | 5.47 |
| Qwen2.5-VL-7B + LoRA v3 | 56.34 | 24.67 | 20.54 |
| Qwen3-VL-8B (Base) | 11.09 | 2.72 | 22.01 |

> SFT v2: full-param, `image_max_pixels` 1M, projector unfrozen.
> MoE v1: LoRA r=16, q/v only, frozen projector, 6 experts (router collapsed).
> LoRA v3: LoRA r=32, 7 modules, frozen projector (PEFT override). For RL MoE init.

## Table 2: Per-Domain Step Success Rate (%)

| Task | Domain | Base | SFT v2 | MoE v1 | LoRA v3 | Paper |
|------|--------|:----:|:------:|:------:|:-------:|:-----:|
| **Grounding** | PPT | 50.72 | **69.39** | 65.37 | 63.26 | — |
| | Word | 44.54 | **72.60** | 63.28 | 59.94 | — |
| | Excel | 31.32 | **68.85** | 51.07 | 44.35 | — |
| | *Overall* | 42.47 | **70.56** | 60.32 | 56.34 | **82.30** |
| **Act. Pred (Vis)** | PPT | 25.65 | **53.17** | 39.51 | 31.65 | — |
| | Word | 16.33 | **45.51** | 32.61 | 23.56 | — |
| | Excel | 13.10 | **42.77** | 27.79 | 19.38 | — |
| | *Overall* | 18.05 | **46.90** | 33.20 | 24.67 | **50.08** |
| **Act. Pred (A11y)** | PPT | 23.57 | 24.22 | 9.01 | **33.02** | — |
| | Word | 16.46 | 22.64 | 5.33 | **23.15** | — |
| | Excel | 2.48 | 2.82 | 2.15 | **4.00** | — |
| | *Overall* | 14.53 | 17.51 | 5.47 | **20.54** | **25.78** |

> **Bold** = best among our models per row. SFT v1 and Qwen3-8B omitted for brevity.

## Table 3: SFT v2 vs Paper — Gap Analysis

| Task | Paper SFT | SFT v2 | Δ (abs) | % of Paper |
|------|:---------:|:------:|:-------:|:----------:|
| Grounding | 82.30 | 70.56 | −11.74 | 85.7% |
| Action Pred (Visual) | 50.08 | 46.90 | −3.18 | 93.7% |
| Action Pred (A11y) | 25.78 | 17.51 | −8.27 | 67.9% |

## Table 4: LoRA v3 vs SFT v2 — Capacity Gap

| Task | SFT v2 (full) | LoRA v3 | Δ | Note |
|------|:-------------:|:-------:|:---:|------|
| Grounding | 70.56 | 56.34 | −14.22 | Frozen projector limits grounding |
| Action Pred (Vis) | 46.90 | 24.67 | −22.23 | Func match 87.5% but coord match only 29.3% |
| Action Pred (A11y) | 17.51 | 20.54 | **+3.03** | LoRA regularization helps A11y |

## Table 5: SFT v2 Relaxed BBox Evaluation (±N px)

| Expand | Grounding | Visual | Paper Ground | Paper Visual |
|:------:|:---------:|:------:|:------------:|:------------:|
| ±0 px | 70.56 | 46.90 | 82.30 | 50.08 |
| ±20 px | 76.01 | **51.81** | — | **50.08** |
| ±50 px | **81.14** | 56.50 | **82.30** | — |

> Gap is coordinate offset, not wrong element. ±50px matches paper grounding.

## Key Findings

1. **SFT v2 (full) best on Visual tasks**: 85–94% of paper. Resolution + projector unfreezing critical.
2. **LoRA v3 A11y surpasses SFT v2**: 20.54% vs 17.51% (+3.03%). Limited capacity regularizes against visual-only overfitting.
3. **LoRA capacity gap on Visual**: −14 to −22% vs SFT v2. Click func match 87.5% but coord precision only 29.3%. Frozen projector is root cause.
4. **MoE v1 router collapse**: Single expert → A11y catastrophic at 5.47%.
5. **Next: RL MoE**: LoRA v3 → copy to N experts → RL reward (action_match + f_pseudo) should improve coord precision and function selection.

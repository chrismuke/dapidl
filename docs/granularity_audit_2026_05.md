# Granularity audit — cell counts by scenario × tier

_Generated from `pipeline_output/granularity_audit/per_slide_counts.parquet`_

**Tier definitions** (single source of truth: `dapidl.ontology.training_tiers`):

- **COARSE (5)** — CL Super-Coarse (Level 1): Epithelial · Immune · Stromal · Endothelial · Neural
- **MEDIUM (12)** — CL Coarse + L3, body-wide: Epithelial_Luminal · Epithelial_Basal · T_Cell · B_Cell · Macrophage · Dendritic_Cell · Mast_Cell · Fibroblast · Pericyte · Adipocyte · Endothelial · Neural
- **FINE (18)** — CL Medium L3+L4 + 2 pathology: CD4_T_Cell · CD8_T_Cell · Treg · B_Cell · Plasma_Cell · NK_Cell · Macrophage · pDC · cDC · Mast_Cell · Mammary_Luminal · Myoepithelial · Fibroblast · Pericyte · Adipocyte · Endothelial · DCIS · Invasive

Cells whose CL ancestry doesn't match a specific class fall back to the parent compartment name (e.g., `keratinocyte` → MEDIUM:`Epithelial`). `Unknown` = no CL mapping at all.

---

## TIER: COARSE  (5 canonical classes)

### Individual scenarios

### STHELAR breast s0

_Total cells: **576,963**_

| # | class | count | % |
|---|-------|------:|----:|
| 1 | Epithelial | 230,669 | 39.98% |
| 2 | Immune | 172,433 | 29.89% |
| 3 | Stromal | 111,606 | 19.34% |
| 4 | Endothelial | 62,255 | 10.79% |
| 5 | Neural | 0 | 0.00% |

### STHELAR breast s1

_Total cells: **892,966**_

| # | class | count | % |
|---|-------|------:|----:|
| 1 | Epithelial | 330,368 | 37.00% |
| 2 | Immune | 297,111 | 33.27% |
| 3 | Stromal | 174,076 | 19.49% |
| 4 | Endothelial | 91,411 | 10.24% |
| 5 | Neural | 0 | 0.00% |

### STHELAR breast s3

_Total cells: **365,604**_

| # | class | count | % |
|---|-------|------:|----:|
| 1 | Epithelial | 159,707 | 43.68% |
| 2 | Immune | 93,009 | 25.44% |
| 3 | Stromal | 71,952 | 19.68% |
| 4 | Endothelial | 40,936 | 11.20% |
| 5 | Neural | 0 | 0.00% |

### STHELAR breast s6 (Prime)

_Total cells: **692,184**_

| # | class | count | % |
|---|-------|------:|----:|
| 1 | Epithelial | 159,184 | 23.00% |
| 2 | Immune | 398,289 | 57.54% |
| 3 | Stromal | 81,302 | 11.75% |
| 4 | Endothelial | 53,409 | 7.72% |
| 5 | Neural | 0 | 0.00% |

### Janesick rep1

_Total cells: **167,780**_

| # | class | count | % |
|---|-------|------:|----:|
| 1 | Epithelial | 72,693 | 43.33% |
| 2 | Immune | 34,137 | 20.35% |
| 3 | Stromal | 42,269 | 25.19% |
| 4 | Endothelial | 8,931 | 5.32% |
| 5 | Neural | 0 | 0.00% |
| 6 | Unknown | 9,750 | 5.81% |

### Janesick rep2

_Total cells: **118,752**_

| # | class | count | % |
|---|-------|------:|----:|
| 1 | Epithelial | 43,514 | 36.64% |
| 2 | Immune | 25,077 | 21.12% |
| 3 | Stromal | 39,193 | 33.00% |
| 4 | Endothelial | 6,699 | 5.64% |
| 5 | Neural | 0 | 0.00% |
| 6 | Unknown | 4,269 | 3.59% |

### Grouped scenarios

### STHELAR breast standard (s0 + s1 + s3)

_Total cells: **1,835,533**_

| # | class | count | % |
|---|-------|------:|----:|
| 1 | Epithelial | 720,744 | 39.27% |
| 2 | Immune | 562,553 | 30.65% |
| 3 | Stromal | 357,634 | 19.48% |
| 4 | Endothelial | 194,602 | 10.60% |
| 5 | Neural | 0 | 0.00% |

### STHELAR breast Prime (s6 only)

_Total cells: **692,184**_

| # | class | count | % |
|---|-------|------:|----:|
| 1 | Epithelial | 159,184 | 23.00% |
| 2 | Immune | 398,289 | 57.54% |
| 3 | Stromal | 81,302 | 11.75% |
| 4 | Endothelial | 53,409 | 7.72% |
| 5 | Neural | 0 | 0.00% |

### Janesick (rep1 + rep2)

_Total cells: **286,532**_

| # | class | count | % |
|---|-------|------:|----:|
| 1 | Epithelial | 116,207 | 40.56% |
| 2 | Immune | 59,214 | 20.67% |
| 3 | Stromal | 81,462 | 28.43% |
| 4 | Endothelial | 15,630 | 5.45% |
| 5 | Neural | 0 | 0.00% |
| 6 | Unknown | 14,019 | 4.89% |

### All STHELAR breast (s0 + s1 + s3 + s6)

_Total cells: **2,527,717**_

| # | class | count | % |
|---|-------|------:|----:|
| 1 | Epithelial | 879,928 | 34.81% |
| 2 | Immune | 960,842 | 38.01% |
| 3 | Stromal | 438,936 | 17.36% |
| 4 | Endothelial | 248,011 | 9.81% |
| 5 | Neural | 0 | 0.00% |

### All breast (STHELAR + Janesick)

_Total cells: **2,814,249**_

| # | class | count | % |
|---|-------|------:|----:|
| 1 | Epithelial | 996,135 | 35.40% |
| 2 | Immune | 1,020,056 | 36.25% |
| 3 | Stromal | 520,398 | 18.49% |
| 4 | Endothelial | 263,641 | 9.37% |
| 5 | Neural | 0 | 0.00% |
| 6 | Unknown | 14,019 | 0.50% |

### All STHELAR skin

_Total cells: **394,380**_

| # | class | count | % |
|---|-------|------:|----:|
| 1 | Epithelial | 57,677 | 14.62% |
| 2 | Immune | 150,849 | 38.25% |
| 3 | Stromal | 111,587 | 28.29% |
| 4 | Endothelial | 61,993 | 15.72% |
| 5 | Neural | 12,274 | 3.11% |

### All STHELAR (all 16 tissues)

_Total cells: **11,207,689**_

| # | class | count | % |
|---|-------|------:|----:|
| 1 | Epithelial | 1,926,116 | 17.19% |
| 2 | Immune | 6,258,096 | 55.84% |
| 3 | Stromal | 1,343,217 | 11.98% |
| 4 | Endothelial | 657,961 | 5.87% |
| 5 | Neural | 926,185 | 8.26% |
| 6 | Unknown | 96,114 | 0.86% |

---

## TIER: MEDIUM  (12 canonical classes)

### Individual scenarios

### STHELAR breast s0

_Total cells: **576,963**_

| # | class | count | % |
|---|-------|------:|----:|
| 1 | Epithelial_Luminal | 152,867 | 26.50% |
| 2 | Epithelial_Basal | 77,802 | 13.48% |
| 3 | T_Cell | 34,964 | 6.06% |
| 4 | B_Cell | 11,179 | 1.94% |
| 5 | Macrophage | 29,845 | 5.17% |
| 6 | Dendritic_Cell | 19,621 | 3.40% |
| 7 | Mast_Cell | 43,151 | 7.48% |
| 8 | Fibroblast | 49,423 | 8.57% |
| 9 | Pericyte | 31,287 | 5.42% |
| 10 | Adipocyte | 0 | 0.00% |
| 11 | Endothelial | 62,255 | 10.79% |
| 12 | Neural | 0 | 0.00% |
| 13 | Immune | 33,673 | 5.84% |
| 14 | Stromal | 30,896 | 5.35% |

### STHELAR breast s1

_Total cells: **892,966**_

| # | class | count | % |
|---|-------|------:|----:|
| 1 | Epithelial_Luminal | 208,949 | 23.40% |
| 2 | Epithelial_Basal | 121,419 | 13.60% |
| 3 | T_Cell | 49,320 | 5.52% |
| 4 | B_Cell | 22,938 | 2.57% |
| 5 | Macrophage | 52,768 | 5.91% |
| 6 | Dendritic_Cell | 79,155 | 8.86% |
| 7 | Mast_Cell | 11,224 | 1.26% |
| 8 | Fibroblast | 82,845 | 9.28% |
| 9 | Pericyte | 46,981 | 5.26% |
| 10 | Adipocyte | 0 | 0.00% |
| 11 | Endothelial | 91,411 | 10.24% |
| 12 | Neural | 0 | 0.00% |
| 13 | Stromal | 44,250 | 4.96% |
| 14 | Immune | 81,706 | 9.15% |

### STHELAR breast s3

_Total cells: **365,604**_

| # | class | count | % |
|---|-------|------:|----:|
| 1 | Epithelial_Luminal | 105,831 | 28.95% |
| 2 | Epithelial_Basal | 53,876 | 14.74% |
| 3 | T_Cell | 20,147 | 5.51% |
| 4 | B_Cell | 6,816 | 1.86% |
| 5 | Macrophage | 17,214 | 4.71% |
| 6 | Dendritic_Cell | 21,729 | 5.94% |
| 7 | Mast_Cell | 5,050 | 1.38% |
| 8 | Fibroblast | 31,630 | 8.65% |
| 9 | Pericyte | 21,293 | 5.82% |
| 10 | Adipocyte | 0 | 0.00% |
| 11 | Endothelial | 40,936 | 11.20% |
| 12 | Neural | 0 | 0.00% |
| 13 | Immune | 22,053 | 6.03% |
| 14 | Stromal | 19,029 | 5.20% |

### STHELAR breast s6 (Prime)

_Total cells: **692,184**_

| # | class | count | % |
|---|-------|------:|----:|
| 1 | Epithelial_Luminal | 85,796 | 12.39% |
| 2 | Epithelial_Basal | 73,388 | 10.60% |
| 3 | T_Cell | 54,146 | 7.82% |
| 4 | B_Cell | 21,167 | 3.06% |
| 5 | Macrophage | 37,287 | 5.39% |
| 6 | Dendritic_Cell | 28,991 | 4.19% |
| 7 | Mast_Cell | 202,128 | 29.20% |
| 8 | Fibroblast | 32,874 | 4.75% |
| 9 | Pericyte | 25,164 | 3.64% |
| 10 | Adipocyte | 0 | 0.00% |
| 11 | Endothelial | 53,409 | 7.72% |
| 12 | Neural | 0 | 0.00% |
| 13 | Stromal | 23,264 | 3.36% |
| 14 | Immune | 54,570 | 7.88% |

### Janesick rep1

_Total cells: **167,780**_

| # | class | count | % |
|---|-------|------:|----:|
| 1 | Epithelial_Luminal | 62,755 | 37.40% |
| 2 | Epithelial_Basal | 0 | 0.00% |
| 3 | T_Cell | 15,393 | 9.17% |
| 4 | B_Cell | 4,987 | 2.97% |
| 5 | Macrophage | 12,798 | 7.63% |
| 6 | Dendritic_Cell | 792 | 0.47% |
| 7 | Mast_Cell | 167 | 0.10% |
| 8 | Fibroblast | 41,422 | 24.69% |
| 9 | Pericyte | 847 | 0.50% |
| 10 | Adipocyte | 0 | 0.00% |
| 11 | Endothelial | 8,931 | 5.32% |
| 12 | Neural | 0 | 0.00% |
| 13 | Epithelial | 9,938 | 5.92% |
| 14 | Unknown | 9,750 | 5.81% |

### Janesick rep2

_Total cells: **118,752**_

| # | class | count | % |
|---|-------|------:|----:|
| 1 | Epithelial_Luminal | 34,805 | 29.31% |
| 2 | Epithelial_Basal | 0 | 0.00% |
| 3 | T_Cell | 10,888 | 9.17% |
| 4 | B_Cell | 3,792 | 3.19% |
| 5 | Macrophage | 9,630 | 8.11% |
| 6 | Dendritic_Cell | 606 | 0.51% |
| 7 | Mast_Cell | 161 | 0.14% |
| 8 | Fibroblast | 38,644 | 32.54% |
| 9 | Pericyte | 549 | 0.46% |
| 10 | Adipocyte | 0 | 0.00% |
| 11 | Endothelial | 6,699 | 5.64% |
| 12 | Neural | 0 | 0.00% |
| 13 | Unknown | 4,269 | 3.59% |
| 14 | Epithelial | 8,709 | 7.33% |

### Grouped scenarios

### STHELAR breast standard (s0 + s1 + s3)

_Total cells: **1,835,533**_

| # | class | count | % |
|---|-------|------:|----:|
| 1 | Epithelial_Luminal | 467,647 | 25.48% |
| 2 | Epithelial_Basal | 253,097 | 13.79% |
| 3 | T_Cell | 104,431 | 5.69% |
| 4 | B_Cell | 40,933 | 2.23% |
| 5 | Macrophage | 99,827 | 5.44% |
| 6 | Dendritic_Cell | 120,505 | 6.57% |
| 7 | Mast_Cell | 59,425 | 3.24% |
| 8 | Fibroblast | 163,898 | 8.93% |
| 9 | Pericyte | 99,561 | 5.42% |
| 10 | Adipocyte | 0 | 0.00% |
| 11 | Endothelial | 194,602 | 10.60% |
| 12 | Neural | 0 | 0.00% |
| 13 | Stromal | 94,175 | 5.13% |
| 14 | Immune | 137,432 | 7.49% |

### STHELAR breast Prime (s6 only)

_Total cells: **692,184**_

| # | class | count | % |
|---|-------|------:|----:|
| 1 | Epithelial_Luminal | 85,796 | 12.39% |
| 2 | Epithelial_Basal | 73,388 | 10.60% |
| 3 | T_Cell | 54,146 | 7.82% |
| 4 | B_Cell | 21,167 | 3.06% |
| 5 | Macrophage | 37,287 | 5.39% |
| 6 | Dendritic_Cell | 28,991 | 4.19% |
| 7 | Mast_Cell | 202,128 | 29.20% |
| 8 | Fibroblast | 32,874 | 4.75% |
| 9 | Pericyte | 25,164 | 3.64% |
| 10 | Adipocyte | 0 | 0.00% |
| 11 | Endothelial | 53,409 | 7.72% |
| 12 | Neural | 0 | 0.00% |
| 13 | Stromal | 23,264 | 3.36% |
| 14 | Immune | 54,570 | 7.88% |

### Janesick (rep1 + rep2)

_Total cells: **286,532**_

| # | class | count | % |
|---|-------|------:|----:|
| 1 | Epithelial_Luminal | 97,560 | 34.05% |
| 2 | Epithelial_Basal | 0 | 0.00% |
| 3 | T_Cell | 26,281 | 9.17% |
| 4 | B_Cell | 8,779 | 3.06% |
| 5 | Macrophage | 22,428 | 7.83% |
| 6 | Dendritic_Cell | 1,398 | 0.49% |
| 7 | Mast_Cell | 328 | 0.11% |
| 8 | Fibroblast | 80,066 | 27.94% |
| 9 | Pericyte | 1,396 | 0.49% |
| 10 | Adipocyte | 0 | 0.00% |
| 11 | Endothelial | 15,630 | 5.45% |
| 12 | Neural | 0 | 0.00% |
| 13 | Epithelial | 18,647 | 6.51% |
| 14 | Unknown | 14,019 | 4.89% |

### All STHELAR breast (s0 + s1 + s3 + s6)

_Total cells: **2,527,717**_

| # | class | count | % |
|---|-------|------:|----:|
| 1 | Epithelial_Luminal | 553,443 | 21.89% |
| 2 | Epithelial_Basal | 326,485 | 12.92% |
| 3 | T_Cell | 158,577 | 6.27% |
| 4 | B_Cell | 62,100 | 2.46% |
| 5 | Macrophage | 137,114 | 5.42% |
| 6 | Dendritic_Cell | 149,496 | 5.91% |
| 7 | Mast_Cell | 261,553 | 10.35% |
| 8 | Fibroblast | 196,772 | 7.78% |
| 9 | Pericyte | 124,725 | 4.93% |
| 10 | Adipocyte | 0 | 0.00% |
| 11 | Endothelial | 248,011 | 9.81% |
| 12 | Neural | 0 | 0.00% |
| 13 | Stromal | 117,439 | 4.65% |
| 14 | Immune | 192,002 | 7.60% |

### All breast (STHELAR + Janesick)

_Total cells: **2,814,249**_

| # | class | count | % |
|---|-------|------:|----:|
| 1 | Epithelial_Luminal | 651,003 | 23.13% |
| 2 | Epithelial_Basal | 326,485 | 11.60% |
| 3 | T_Cell | 184,858 | 6.57% |
| 4 | B_Cell | 70,879 | 2.52% |
| 5 | Macrophage | 159,542 | 5.67% |
| 6 | Dendritic_Cell | 150,894 | 5.36% |
| 7 | Mast_Cell | 261,881 | 9.31% |
| 8 | Fibroblast | 276,838 | 9.84% |
| 9 | Pericyte | 126,121 | 4.48% |
| 10 | Adipocyte | 0 | 0.00% |
| 11 | Endothelial | 263,641 | 9.37% |
| 12 | Neural | 0 | 0.00% |
| 13 | Unknown | 14,019 | 0.50% |
| 14 | Epithelial | 18,647 | 0.66% |
| 15 | Immune | 192,002 | 6.82% |
| 16 | Stromal | 117,439 | 4.17% |

### All STHELAR skin

_Total cells: **394,380**_

| # | class | count | % |
|---|-------|------:|----:|
| 1 | Epithelial_Luminal | 0 | 0.00% |
| 2 | Epithelial_Basal | 0 | 0.00% |
| 3 | T_Cell | 58,518 | 14.84% |
| 4 | B_Cell | 19,047 | 4.83% |
| 5 | Macrophage | 7,836 | 1.99% |
| 6 | Dendritic_Cell | 27,786 | 7.05% |
| 7 | Mast_Cell | 17,155 | 4.35% |
| 8 | Fibroblast | 88,880 | 22.54% |
| 9 | Pericyte | 9,822 | 2.49% |
| 10 | Adipocyte | 0 | 0.00% |
| 11 | Endothelial | 61,993 | 15.72% |
| 12 | Neural | 12,274 | 3.11% |
| 13 | Stromal | 12,885 | 3.27% |
| 14 | Immune | 20,507 | 5.20% |
| 15 | Epithelial | 57,677 | 14.62% |

### All STHELAR (all 16 tissues)

_Total cells: **11,207,689**_

| # | class | count | % |
|---|-------|------:|----:|
| 1 | Epithelial_Luminal | 553,443 | 4.94% |
| 2 | Epithelial_Basal | 329,354 | 2.94% |
| 3 | T_Cell | 1,578,895 | 14.09% |
| 4 | B_Cell | 1,334,309 | 11.91% |
| 5 | Macrophage | 734,190 | 6.55% |
| 6 | Dendritic_Cell | 922,701 | 8.23% |
| 7 | Mast_Cell | 344,231 | 3.07% |
| 8 | Fibroblast | 806,849 | 7.20% |
| 9 | Pericyte | 289,741 | 2.59% |
| 10 | Adipocyte | 531 | 0.00% |
| 11 | Endothelial | 657,961 | 5.87% |
| 12 | Neural | 926,185 | 8.26% |
| 13 | Epithelial | 1,043,319 | 9.31% |
| 14 | Unknown | 96,114 | 0.86% |
| 15 | Stromal | 246,096 | 2.20% |
| 16 | Immune | 1,343,770 | 11.99% |

---

## TIER: FINE  (18 canonical classes)

### Individual scenarios

### STHELAR breast s0

_Total cells: **576,963**_

| # | class | count | % |
|---|-------|------:|----:|
| 1 | CD4_T_Cell | 17,173 | 2.98% |
| 2 | CD8_T_Cell | 13,720 | 2.38% |
| 3 | Treg | 4,071 | 0.71% |
| 4 | B_Cell | 6,548 | 1.13% |
| 5 | Plasma_Cell | 4,631 | 0.80% |
| 6 | NK_Cell | 6,049 | 1.05% |
| 7 | Macrophage | 29,845 | 5.17% |
| 8 | pDC | 10,679 | 1.85% |
| 9 | cDC | 0 | 0.00% |
| 10 | Mast_Cell | 43,151 | 7.48% |
| 11 | Mammary_Luminal | 152,867 | 26.50% |
| 12 | Myoepithelial | 0 | 0.00% |
| 13 | Fibroblast | 49,423 | 8.57% |
| 14 | Pericyte | 31,287 | 5.42% |
| 15 | Adipocyte | 0 | 0.00% |
| 16 | Endothelial | 62,255 | 10.79% |
| 17 | DCIS | 0 | 0.00% |
| 18 | Invasive | 0 | 0.00% |
| 19 | Epithelial | 77,802 | 13.48% |
| 20 | Immune | 36,566 | 6.34% |
| 21 | Stromal | 30,896 | 5.35% |

### STHELAR breast s1

_Total cells: **892,966**_

| # | class | count | % |
|---|-------|------:|----:|
| 1 | CD4_T_Cell | 19,052 | 2.13% |
| 2 | CD8_T_Cell | 21,186 | 2.37% |
| 3 | Treg | 9,082 | 1.02% |
| 4 | B_Cell | 12,831 | 1.44% |
| 5 | Plasma_Cell | 10,107 | 1.13% |
| 6 | NK_Cell | 38,791 | 4.34% |
| 7 | Macrophage | 52,768 | 5.91% |
| 8 | pDC | 26,480 | 2.97% |
| 9 | cDC | 0 | 0.00% |
| 10 | Mast_Cell | 11,224 | 1.26% |
| 11 | Mammary_Luminal | 208,949 | 23.40% |
| 12 | Myoepithelial | 0 | 0.00% |
| 13 | Fibroblast | 82,845 | 9.28% |
| 14 | Pericyte | 46,981 | 5.26% |
| 15 | Adipocyte | 0 | 0.00% |
| 16 | Endothelial | 91,411 | 10.24% |
| 17 | DCIS | 0 | 0.00% |
| 18 | Invasive | 0 | 0.00% |
| 19 | Immune | 95,590 | 10.70% |
| 20 | Epithelial | 121,419 | 13.60% |
| 21 | Stromal | 44,250 | 4.96% |

### STHELAR breast s3

_Total cells: **365,604**_

| # | class | count | % |
|---|-------|------:|----:|
| 1 | CD4_T_Cell | 8,898 | 2.43% |
| 2 | CD8_T_Cell | 8,637 | 2.36% |
| 3 | Treg | 2,612 | 0.71% |
| 4 | B_Cell | 3,564 | 0.97% |
| 5 | Plasma_Cell | 3,252 | 0.89% |
| 6 | NK_Cell | 5,576 | 1.53% |
| 7 | Macrophage | 17,214 | 4.71% |
| 8 | pDC | 10,325 | 2.82% |
| 9 | cDC | 0 | 0.00% |
| 10 | Mast_Cell | 5,050 | 1.38% |
| 11 | Mammary_Luminal | 105,831 | 28.95% |
| 12 | Myoepithelial | 0 | 0.00% |
| 13 | Fibroblast | 31,630 | 8.65% |
| 14 | Pericyte | 21,293 | 5.82% |
| 15 | Adipocyte | 0 | 0.00% |
| 16 | Endothelial | 40,936 | 11.20% |
| 17 | DCIS | 0 | 0.00% |
| 18 | Invasive | 0 | 0.00% |
| 19 | Immune | 27,881 | 7.63% |
| 20 | Stromal | 19,029 | 5.20% |
| 21 | Epithelial | 53,876 | 14.74% |

### STHELAR breast s6 (Prime)

_Total cells: **692,184**_

| # | class | count | % |
|---|-------|------:|----:|
| 1 | CD4_T_Cell | 21,020 | 3.04% |
| 2 | CD8_T_Cell | 22,865 | 3.30% |
| 3 | Treg | 10,261 | 1.48% |
| 4 | B_Cell | 11,771 | 1.70% |
| 5 | Plasma_Cell | 9,396 | 1.36% |
| 6 | NK_Cell | 22,452 | 3.24% |
| 7 | Macrophage | 37,287 | 5.39% |
| 8 | pDC | 14,437 | 2.09% |
| 9 | cDC | 0 | 0.00% |
| 10 | Mast_Cell | 202,128 | 29.20% |
| 11 | Mammary_Luminal | 85,796 | 12.39% |
| 12 | Myoepithelial | 0 | 0.00% |
| 13 | Fibroblast | 32,874 | 4.75% |
| 14 | Pericyte | 25,164 | 3.64% |
| 15 | Adipocyte | 0 | 0.00% |
| 16 | Endothelial | 53,409 | 7.72% |
| 17 | DCIS | 0 | 0.00% |
| 18 | Invasive | 0 | 0.00% |
| 19 | Immune | 46,672 | 6.74% |
| 20 | Stromal | 23,264 | 3.36% |
| 21 | Epithelial | 73,388 | 10.60% |

### Janesick rep1

_Total cells: **167,780**_

| # | class | count | % |
|---|-------|------:|----:|
| 1 | CD4_T_Cell | 8,453 | 5.04% |
| 2 | CD8_T_Cell | 6,940 | 4.14% |
| 3 | Treg | 0 | 0.00% |
| 4 | B_Cell | 4,987 | 2.97% |
| 5 | Plasma_Cell | 0 | 0.00% |
| 6 | NK_Cell | 0 | 0.00% |
| 7 | Macrophage | 12,798 | 7.63% |
| 8 | pDC | 494 | 0.29% |
| 9 | cDC | 0 | 0.00% |
| 10 | Mast_Cell | 167 | 0.10% |
| 11 | Mammary_Luminal | 0 | 0.00% |
| 12 | Myoepithelial | 9,938 | 5.92% |
| 13 | Fibroblast | 41,422 | 24.69% |
| 14 | Pericyte | 847 | 0.50% |
| 15 | Adipocyte | 0 | 0.00% |
| 16 | Endothelial | 8,931 | 5.32% |
| 17 | DCIS | 24,606 | 14.67% |
| 18 | Invasive | 38,149 | 22.74% |
| 19 | Unknown | 9,750 | 5.81% |
| 20 | Immune | 298 | 0.18% |

### Janesick rep2

_Total cells: **118,752**_

| # | class | count | % |
|---|-------|------:|----:|
| 1 | CD4_T_Cell | 5,940 | 5.00% |
| 2 | CD8_T_Cell | 4,948 | 4.17% |
| 3 | Treg | 0 | 0.00% |
| 4 | B_Cell | 3,792 | 3.19% |
| 5 | Plasma_Cell | 0 | 0.00% |
| 6 | NK_Cell | 0 | 0.00% |
| 7 | Macrophage | 9,630 | 8.11% |
| 8 | pDC | 400 | 0.34% |
| 9 | cDC | 0 | 0.00% |
| 10 | Mast_Cell | 161 | 0.14% |
| 11 | Mammary_Luminal | 0 | 0.00% |
| 12 | Myoepithelial | 8,709 | 7.33% |
| 13 | Fibroblast | 38,644 | 32.54% |
| 14 | Pericyte | 549 | 0.46% |
| 15 | Adipocyte | 0 | 0.00% |
| 16 | Endothelial | 6,699 | 5.64% |
| 17 | DCIS | 17,195 | 14.48% |
| 18 | Invasive | 17,610 | 14.83% |
| 19 | Immune | 206 | 0.17% |
| 20 | Unknown | 4,269 | 3.59% |

### Grouped scenarios

### STHELAR breast standard (s0 + s1 + s3)

_Total cells: **1,835,533**_

| # | class | count | % |
|---|-------|------:|----:|
| 1 | CD4_T_Cell | 45,123 | 2.46% |
| 2 | CD8_T_Cell | 43,543 | 2.37% |
| 3 | Treg | 15,765 | 0.86% |
| 4 | B_Cell | 22,943 | 1.25% |
| 5 | Plasma_Cell | 17,990 | 0.98% |
| 6 | NK_Cell | 50,416 | 2.75% |
| 7 | Macrophage | 99,827 | 5.44% |
| 8 | pDC | 47,484 | 2.59% |
| 9 | cDC | 0 | 0.00% |
| 10 | Mast_Cell | 59,425 | 3.24% |
| 11 | Mammary_Luminal | 467,647 | 25.48% |
| 12 | Myoepithelial | 0 | 0.00% |
| 13 | Fibroblast | 163,898 | 8.93% |
| 14 | Pericyte | 99,561 | 5.42% |
| 15 | Adipocyte | 0 | 0.00% |
| 16 | Endothelial | 194,602 | 10.60% |
| 17 | DCIS | 0 | 0.00% |
| 18 | Invasive | 0 | 0.00% |
| 19 | Stromal | 94,175 | 5.13% |
| 20 | Epithelial | 253,097 | 13.79% |
| 21 | Immune | 160,037 | 8.72% |

### STHELAR breast Prime (s6 only)

_Total cells: **692,184**_

| # | class | count | % |
|---|-------|------:|----:|
| 1 | CD4_T_Cell | 21,020 | 3.04% |
| 2 | CD8_T_Cell | 22,865 | 3.30% |
| 3 | Treg | 10,261 | 1.48% |
| 4 | B_Cell | 11,771 | 1.70% |
| 5 | Plasma_Cell | 9,396 | 1.36% |
| 6 | NK_Cell | 22,452 | 3.24% |
| 7 | Macrophage | 37,287 | 5.39% |
| 8 | pDC | 14,437 | 2.09% |
| 9 | cDC | 0 | 0.00% |
| 10 | Mast_Cell | 202,128 | 29.20% |
| 11 | Mammary_Luminal | 85,796 | 12.39% |
| 12 | Myoepithelial | 0 | 0.00% |
| 13 | Fibroblast | 32,874 | 4.75% |
| 14 | Pericyte | 25,164 | 3.64% |
| 15 | Adipocyte | 0 | 0.00% |
| 16 | Endothelial | 53,409 | 7.72% |
| 17 | DCIS | 0 | 0.00% |
| 18 | Invasive | 0 | 0.00% |
| 19 | Stromal | 23,264 | 3.36% |
| 20 | Epithelial | 73,388 | 10.60% |
| 21 | Immune | 46,672 | 6.74% |

### Janesick (rep1 + rep2)

_Total cells: **286,532**_

| # | class | count | % |
|---|-------|------:|----:|
| 1 | CD4_T_Cell | 14,393 | 5.02% |
| 2 | CD8_T_Cell | 11,888 | 4.15% |
| 3 | Treg | 0 | 0.00% |
| 4 | B_Cell | 8,779 | 3.06% |
| 5 | Plasma_Cell | 0 | 0.00% |
| 6 | NK_Cell | 0 | 0.00% |
| 7 | Macrophage | 22,428 | 7.83% |
| 8 | pDC | 894 | 0.31% |
| 9 | cDC | 0 | 0.00% |
| 10 | Mast_Cell | 328 | 0.11% |
| 11 | Mammary_Luminal | 0 | 0.00% |
| 12 | Myoepithelial | 18,647 | 6.51% |
| 13 | Fibroblast | 80,066 | 27.94% |
| 14 | Pericyte | 1,396 | 0.49% |
| 15 | Adipocyte | 0 | 0.00% |
| 16 | Endothelial | 15,630 | 5.45% |
| 17 | DCIS | 41,801 | 14.59% |
| 18 | Invasive | 55,759 | 19.46% |
| 19 | Unknown | 14,019 | 4.89% |
| 20 | Immune | 504 | 0.18% |

### All STHELAR breast (s0 + s1 + s3 + s6)

_Total cells: **2,527,717**_

| # | class | count | % |
|---|-------|------:|----:|
| 1 | CD4_T_Cell | 66,143 | 2.62% |
| 2 | CD8_T_Cell | 66,408 | 2.63% |
| 3 | Treg | 26,026 | 1.03% |
| 4 | B_Cell | 34,714 | 1.37% |
| 5 | Plasma_Cell | 27,386 | 1.08% |
| 6 | NK_Cell | 72,868 | 2.88% |
| 7 | Macrophage | 137,114 | 5.42% |
| 8 | pDC | 61,921 | 2.45% |
| 9 | cDC | 0 | 0.00% |
| 10 | Mast_Cell | 261,553 | 10.35% |
| 11 | Mammary_Luminal | 553,443 | 21.89% |
| 12 | Myoepithelial | 0 | 0.00% |
| 13 | Fibroblast | 196,772 | 7.78% |
| 14 | Pericyte | 124,725 | 4.93% |
| 15 | Adipocyte | 0 | 0.00% |
| 16 | Endothelial | 248,011 | 9.81% |
| 17 | DCIS | 0 | 0.00% |
| 18 | Invasive | 0 | 0.00% |
| 19 | Epithelial | 326,485 | 12.92% |
| 20 | Stromal | 117,439 | 4.65% |
| 21 | Immune | 206,709 | 8.18% |

### All breast (STHELAR + Janesick)

_Total cells: **2,814,249**_

| # | class | count | % |
|---|-------|------:|----:|
| 1 | CD4_T_Cell | 80,536 | 2.86% |
| 2 | CD8_T_Cell | 78,296 | 2.78% |
| 3 | Treg | 26,026 | 0.92% |
| 4 | B_Cell | 43,493 | 1.55% |
| 5 | Plasma_Cell | 27,386 | 0.97% |
| 6 | NK_Cell | 72,868 | 2.59% |
| 7 | Macrophage | 159,542 | 5.67% |
| 8 | pDC | 62,815 | 2.23% |
| 9 | cDC | 0 | 0.00% |
| 10 | Mast_Cell | 261,881 | 9.31% |
| 11 | Mammary_Luminal | 553,443 | 19.67% |
| 12 | Myoepithelial | 18,647 | 0.66% |
| 13 | Fibroblast | 276,838 | 9.84% |
| 14 | Pericyte | 126,121 | 4.48% |
| 15 | Adipocyte | 0 | 0.00% |
| 16 | Endothelial | 263,641 | 9.37% |
| 17 | DCIS | 41,801 | 1.49% |
| 18 | Invasive | 55,759 | 1.98% |
| 19 | Stromal | 117,439 | 4.17% |
| 20 | Epithelial | 326,485 | 11.60% |
| 21 | Immune | 207,213 | 7.36% |
| 22 | Unknown | 14,019 | 0.50% |

### All STHELAR skin

_Total cells: **394,380**_

| # | class | count | % |
|---|-------|------:|----:|
| 1 | CD4_T_Cell | 14,514 | 3.68% |
| 2 | CD8_T_Cell | 16,677 | 4.23% |
| 3 | Treg | 10,252 | 2.60% |
| 4 | B_Cell | 10,946 | 2.78% |
| 5 | Plasma_Cell | 8,101 | 2.05% |
| 6 | NK_Cell | 8,357 | 2.12% |
| 7 | Macrophage | 7,836 | 1.99% |
| 8 | pDC | 5,492 | 1.39% |
| 9 | cDC | 16,381 | 4.15% |
| 10 | Mast_Cell | 17,155 | 4.35% |
| 11 | Mammary_Luminal | 0 | 0.00% |
| 12 | Myoepithelial | 0 | 0.00% |
| 13 | Fibroblast | 88,880 | 22.54% |
| 14 | Pericyte | 9,822 | 2.49% |
| 15 | Adipocyte | 0 | 0.00% |
| 16 | Endothelial | 61,993 | 15.72% |
| 17 | DCIS | 0 | 0.00% |
| 18 | Invasive | 0 | 0.00% |
| 19 | Neural | 12,274 | 3.11% |
| 20 | Stromal | 12,885 | 3.27% |
| 21 | Immune | 35,138 | 8.91% |
| 22 | Epithelial | 57,677 | 14.62% |

### All STHELAR (all 16 tissues)

_Total cells: **11,207,689**_

| # | class | count | % |
|---|-------|------:|----:|
| 1 | CD4_T_Cell | 511,914 | 4.57% |
| 2 | CD8_T_Cell | 603,888 | 5.39% |
| 3 | Treg | 207,836 | 1.85% |
| 4 | B_Cell | 1,114,789 | 9.95% |
| 5 | Plasma_Cell | 219,520 | 1.96% |
| 6 | NK_Cell | 495,700 | 4.42% |
| 7 | Macrophage | 734,190 | 6.55% |
| 8 | pDC | 204,886 | 1.83% |
| 9 | cDC | 326,646 | 2.91% |
| 10 | Mast_Cell | 344,231 | 3.07% |
| 11 | Mammary_Luminal | 553,443 | 4.94% |
| 12 | Myoepithelial | 0 | 0.00% |
| 13 | Fibroblast | 806,849 | 7.20% |
| 14 | Pericyte | 289,741 | 2.59% |
| 15 | Adipocyte | 531 | 0.00% |
| 16 | Endothelial | 657,961 | 5.87% |
| 17 | DCIS | 0 | 0.00% |
| 18 | Invasive | 0 | 0.00% |
| 19 | Stromal | 246,096 | 2.20% |
| 20 | Unknown | 96,114 | 0.86% |
| 21 | Epithelial | 1,372,673 | 12.25% |
| 22 | Immune | 1,494,496 | 13.33% |
| 23 | Neural | 926,185 | 8.26% |

---

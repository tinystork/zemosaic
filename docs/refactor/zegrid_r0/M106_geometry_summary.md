# M106 — header-only geometry measurements

66 real Seestar S50 sources; canvas 2403 × 3278. All numbers below are geometric, not timing or scientific support.

| Layout | Halo px | Cells nonempty/total | Cell W×H px | Cell N min/median/mean/max | Patch N min/median/mean/max | Cell area | Patch area | Halo overhead | Source crop pixels | Redundancy | Full-source/crop ratio | bbox false positives |
|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2×2 | 0 | 4/4 | 1201–1202 × 1639–1639 | 60.00/61.00/61.00/62.00 | 60.00/61.00/61.00/62.00 | 7877034 | 7877034 | 0.00% | 139023090 | 1.016 | 3.639 | 0 |
| 2×2 | 8 | 4/4 | 1201–1202 × 1639–1639 | 60.00/61.00/61.00/62.00 | 60.00/61.00/61.00/62.00 | 7877034 | 7968186 | 1.16% | 141925856 | 1.037 | 3.565 | 0 |
| 2×2 | 32 | 4/4 | 1201–1202 × 1639–1639 | 60.00/61.00/61.00/62.00 | 60.00/61.00/61.00/62.00 | 7877034 | 8244714 | 4.67% | 150804153 | 1.102 | 3.355 | 0 |
| 5×4 | 0 | 20/20 | 480–481 × 819–820 | 5.00/31.50/34.65/66.00 | 5.00/31.50/34.65/66.00 | 7877034 | 7877034 | 0.00% | 141204867 | 1.032 | 10.177 | 2 |
| 5×4 | 8 | 20/20 | 480–481 × 819–820 | 5.00/31.50/34.65/66.00 | 7.00/32.00/35.00/66.00 | 7877034 | 8205242 | 4.17% | 148561325 | 1.086 | 9.771 | 5 |
| 5×4 | 32 | 20/20 | 480–481 × 819–820 | 5.00/31.50/34.65/66.00 | 7.00/33.00/35.55/66.00 | 7877034 | 9226730 | 17.13% | 171710490 | 1.255 | 8.586 | 0 |
| 7×5 | 0 | 33/35 | 343–344 × 655–656 | 0.00/26.00/28.09/66.00 | 0.00/26.00/28.09/66.00 | 7877034 | 7877034 | 0.00% | 142143330 | 1.039 | 14.340 | 6 |
| 7×5 | 8 | 33/35 | 343–344 × 655–656 | 0.00/26.00/28.09/66.00 | 0.00/26.00/28.49/66.00 | 7877034 | 8351658 | 6.03% | 151755291 | 1.109 | 13.623 | 7 |
| 7×5 | 32 | 33/35 | 343–344 × 655–656 | 0.00/26.00/28.09/66.00 | 1.00/35.00/32.46/66.00 | 7877034 | 9849258 | 25.04% | 182779720 | 1.336 | 12.888 | 12 |
| 9×7 | 0 | 60/63 | 267–267 × 468–469 | 0.00/26.00/28.14/66.00 | 0.00/26.00/28.14/66.00 | 7877034 | 7877034 | 0.00% | 143699403 | 1.050 | 25.585 | 7 |
| 9×7 | 8 | 60/63 | 267–267 × 468–469 | 0.00/26.00/28.14/66.00 | 0.00/26.00/28.40/66.00 | 7877034 | 8539594 | 8.41% | 157428583 | 1.150 | 23.564 | 17 |
| 9×7 | 32 | 60/63 | 267–267 × 468–469 | 0.00/26.00/28.14/66.00 | 0.00/26.00/29.56/66.00 | 7877034 | 10674730 | 35.52% | 201722207 | 1.474 | 19.140 | 10 |

Geometric union covers 84.6552% of canvas. Median projected source bbox = 1091.200 × 1926.268 px.

Counts include empty cells. Empty Cell can still have nonempty halo. Source pixels = spatial pixels (not RGB samples/bytes). Crop margin = 2 source pixels, experimental. Redundancy = sum crop rectangle areas / union of crop rectangles per source; full-source/crop ratio compares reading each full source for the SAME patch memberships, NOT the old production layout or elapsed time.

Complete coverage histograms (80×80 regular sample centres), per-cell coverage-fraction statistics, per-cell memberships, WCS, source header hashes and versions are in M106_geometry.json. Boundary touching with zero area is excluded.

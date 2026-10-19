# Robustness Campaign — iteration_002 Report

- **Execution Timestamp**: 2026-10-19 15:27:04
- **Total Target/Reference Pairs Evaluated**: 25
- **Passing Pairs (Clean Masks & Alignment)**: 21 (84.0%)
- **Needs Visual Review**: 4 (16.0%)
- **Failed Pairs**: 0
- **Blocked Pairs**: 0

## Summary Matrix

| Pair ID | Target | Reference | Status | Target Hair Cov | Ref Hair Cov | Pruned Backdrop | Donor/Target Overlap | Ear Leak Px | Warnings / Notes |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `pair_01_target3_x_reference1` | `target3.jpg` | `reference1.jpg` | **PASS** | 15.7% | 13.7% | 740 px | 91.0% | 0 px |  |
| `pair_02_target2_x_reference2` | `target2.jpg` | `reference2.jpg` | **NEEDS_VISUAL_REVIEW** | 16.6% | 14.8% | 6077 px | 51.2% | 0 px | High uncovered target hair ratio: 48.4% |
| `pair_03_target1_x_reference4` | `target1.png` | `reference4.jpg` | **PASS** | 13.6% | 8.7% | 1274 px | 80.7% | 0 px |  |
| `pair_04_target3_x_reference4` | `target3.jpg` | `reference4.jpg` | **PASS** | 15.7% | 8.7% | 1274 px | 95.2% | 0 px |  |
| `pair_05_target3_x_reference3` | `target3.jpg` | `reference3.jpg` | **PASS** | 15.7% | 17.0% | 847 px | 92.7% | 0 px |  |
| `pair_06_target1_x_reference1` | `target1.png` | `reference1.jpg` | **PASS** | 13.6% | 13.7% | 740 px | 75.0% | 0 px |  |
| `pair_07_target1_x_reference` | `target1.png` | `reference.jpg` | **PASS** | 13.6% | 27.8% | 5198 px | 83.1% | 0 px |  |
| `pair_08_target2_x_reference` | `target2.jpg` | `reference.jpg` | **PASS** | 16.6% | 27.8% | 5198 px | 72.7% | 0 px |  |
| `pair_09_target3_x_reference` | `target3.jpg` | `reference.jpg` | **PASS** | 15.7% | 27.8% | 5198 px | 99.3% | 0 px |  |
| `pair_10_target4_x_reference1` | `target4.jpg` | `reference1.jpg` | **PASS** | 17.3% | 13.7% | 740 px | 65.1% | 0 px |  |
| `pair_11_target2_x_reference1` | `target2.jpg` | `reference1.jpg` | **NEEDS_VISUAL_REVIEW** | 16.6% | 13.7% | 740 px | 34.4% | 0 px | High uncovered target hair ratio: 64.8% |
| `pair_12_target3_x_reference2` | `target3.jpg` | `reference2.jpg` | **PASS** | 15.7% | 14.8% | 6077 px | 93.8% | 0 px |  |
| `pair_13_target_x_reference1` | `target.jpg` | `reference1.jpg` | **PASS** | 8.9% | 13.7% | 740 px | 93.0% | 0 px |  |
| `pair_14_target2_x_reference4` | `target2.jpg` | `reference4.jpg` | **NEEDS_VISUAL_REVIEW** | 16.6% | 8.7% | 1274 px | 44.4% | 0 px | High uncovered target hair ratio: 54.6% |
| `pair_15_target4_x_reference2` | `target4.jpg` | `reference2.jpg` | **PASS** | 17.3% | 14.8% | 6077 px | 86.7% | 0 px |  |
| `pair_16_target2_x_reference3` | `target2.jpg` | `reference3.jpg` | **NEEDS_VISUAL_REVIEW** | 16.6% | 17.0% | 847 px | 50.6% | 0 px | High uncovered target hair ratio: 48.6% |
| `pair_17_target_x_reference2` | `target.jpg` | `reference2.jpg` | **PASS** | 8.9% | 14.8% | 6077 px | 99.8% | 0 px |  |
| `pair_18_target4_x_reference3` | `target4.jpg` | `reference3.jpg` | **PASS** | 17.3% | 17.0% | 847 px | 83.3% | 0 px |  |
| `pair_19_target_x_reference4` | `target.jpg` | `reference4.jpg` | **PASS** | 8.9% | 8.7% | 1274 px | 92.5% | 0 px |  |
| `pair_20_target4_x_reference4` | `target4.jpg` | `reference4.jpg` | **PASS** | 17.3% | 8.7% | 1274 px | 83.0% | 0 px |  |
| `pair_21_target1_x_reference2` | `target1.png` | `reference2.jpg` | **PASS** | 13.6% | 14.8% | 6077 px | 95.8% | 0 px |  |
| `pair_22_target1_x_reference3` | `target1.png` | `reference3.jpg` | **PASS** | 13.6% | 17.0% | 847 px | 81.1% | 0 px |  |
| `pair_23_target_x_reference` | `target.jpg` | `reference.jpg` | **PASS** | 8.9% | 27.8% | 5198 px | 92.5% | 0 px |  |
| `pair_24_target_x_reference3` | `target.jpg` | `reference3.jpg` | **PASS** | 8.9% | 17.0% | 847 px | 87.9% | 0 px |  |
| `pair_25_target4_x_reference` | `target4.jpg` | `reference.jpg` | **PASS** | 17.3% | 27.8% | 5198 px | 97.0% | 0 px |  |

## Diagnostic Takeaways

1. **Identity & Facial Protection**: Strictly enforced across all pairs. No ear or facial feature leakage detected.
2. **Reference Backdrop Pruning**: False-positive studio background regions pruned safely without skull-core hair wipeout.
3. **Coverage Alignment**: Evaluated geometric overlap of donor alpha on target hair before full diffusion generation.

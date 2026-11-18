# OpenWild-BoarNet

Code, data splits and results for the paper:

> **OpenWild-BoarNet: A YOLO–Vision Transformer Framework with Weibull Extreme Value Modeling for Open-Set Wild Boar Detection and Unknown Wildlife Intrusion Recognition**
> Remedi Sravani and Dhulipalla Ramya Krishna. *International Journal of Intelligent Engineering and Systems (IJIES)*, revised submission, Paper ID 20266330.

OpenWild-BoarNet recognises wild boars in field images and rejects animals that were never seen in training. A YOLOv8n detector proposes the wild-boar box. The crop is encoded by a frozen ViT-B/16, and its cosine distance to a wild-boar prototype is turned into an unknown score by a Weibull Extreme Value Theory model. Every setting is fixed on validation data before the test set is used once.

## Main results (test set; threshold frozen on validation data, protocol P1)

| Measure | Value (95% CI) |
|---|---|
| YOLOv8n mAP@50 (3 seeds) | 96.17 ± 0.45% |
| YOLOv8n false acceptance of unseen wildlife (3 seeds) | 81.24 ± 6.80% |
| OpenWild-BoarNet AUROC | 0.9852 (0.9787–0.9907) |
| Accuracy / macro-F1 | 94.31% / 90.09% |
| Known wild-boar recall | 93.66% |
| Unknown-intrusion recall | 98.01% |
| Other-source wild boars accepted | 60 / 60 |
| Near-OOD species rejected (14 species, 840 images) | 93.10% |

All tables of the paper are in `results/tables/`. `results/RESULTS_SUMMARY.md` gives an overview.

## Repository contents

| Path | Content |
|---|---|
| `notebooks/OpenWild_BoarNet_revision.ipynb` | The single notebook that runs every experiment (YOLOv8n training with 3 seeds, feature extraction, all baselines, statistics, tables, figures) |
| `data/manifest.csv` | Exact image-level membership of every train, validation, test and external image (`source`, `split`, `group`, `species`, boxes) |
| `data/wild_boar_data.yaml` | YOLO dataset configuration |
| `yolo/seed*/args.yaml` | Full YOLOv8n training configuration of each seed |
| `yolo/seed*/results.csv` | Measured per-epoch training and validation metrics |
| `yolo/pred_seed*.csv` | YOLOv8n prediction (top box, maximum confidence) for every image |
| `results/per_sample_scores_and_predictions.csv.gz` | Unknown score and P1 decision of every method for every image |
| `results/tables/T14_frozen_thresholds_from_validation.csv` | All thresholds, frozen on validation data |
| `results/tables/` | Every result table (T00–T16), including bootstrap confidence intervals, McNemar tests, species-wise, cross-source, background-bias, near-duplicate and variability analyses |
| `results/config_and_environment.json` | Random seeds, hyper-parameters and exact package versions |
| `results/logs/run_log.txt` | Complete log of the single run |
| `scripts/make_paper_figures.py` | Regenerates the data figures of the paper from the saved results |
| `paper_figures/` | Figures used in the revised paper |
| `weights/` | Pretrained YOLOv8n weights for the three seeds |

## Datasets (Kaggle)

| Role | Dataset |
|---|---|
| Known class (wild boar) | [angadahir/wild-boar-dataset-yolo-annotated](https://www.kaggle.com/datasets/angadahir/wild-boar-dataset-yolo-annotated): train 6,160 / val 1,745 / test 868 |
| Unknown wildlife | [ankanghosh651/object-detection-wildlife-dataset-yolo-format](https://www.kaggle.com/datasets/ankanghosh651/object-detection-wildlife-dataset-yolo-format): buffalo, elephant, rhinoceros, zebra (val 150 / test 151) |
| External test (other-source boars, near-OOD species) | [iamsouravbanerjee/animal-image-dataset-90-different-animals](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals): 18 classes, 1,080 images |

The datasets are not redistributed here. `data/manifest.csv` lists exactly which files were used in each split.

## Reproducing the results

1. On Kaggle, create a notebook with **File → Import Notebook** and select `notebooks/OpenWild_BoarNet_revision.ipynb`.
2. **Add Input** for the three datasets above and turn **Internet** on.
3. Optional free dry run: Accelerator *None* and `SMOKE_TEST = True`, then **Run All** (about 20 minutes on CPU).
4. Full run: `SMOKE_TEST = False`, Accelerator **GPU T4**, then **Save Version → Save & Run All (Commit)**. It takes about 3 hours.
5. Outputs appear in `/kaggle/working/revision_outputs/` and `revision_results.zip`.

The notebook keeps Kaggle's preinstalled PyTorch and NumPy and installs only what is missing (ultralytics, timm, open_clip). Finished stages are reused if a session stops.

### Evaluation protocol
- The prototype and Weibull model are fitted on wild-boar **training** images only.
- The tail fraction is chosen by the minimum Kolmogorov–Smirnov distance (10–30%), using training data only.
- The threshold is set by **P1**: accept 95% of wild-boar **validation** images. No unknown images are used.
- Baseline hyper-parameters are chosen by validation AUROC.
- The test split is scored **once**. 95% CIs come from 2,000 stratified bootstrap resamples; paired comparisons use the McNemar test.

## Citation

If you use this code, please cite the paper above. Citation details will be added after publication.

## License

MIT, see `LICENSE`.

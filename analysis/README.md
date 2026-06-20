# KinConfBench Analysis Pipeline

This folder contains analysis scripts for KinConfBench results. The pipeline is organized into two main branches: **kincore label analysis** (single systems) and **pair analysis** (apo/holo seen/unseen).

## Data layout

Scripts expect data in the unified kinconfbench layout:

```
kinconfbench_root/
├── ABL1_HUMAN/
│   ├── analysis_csvs/           # {stem}_analysis.csv
│   ├── geometric_analysis_csvs/
│   ├── predictions/
│   ├── rcsb_files/
│   ├── {GENE}_summary.csv
│   └── {GENE}_allpdb_summary.csv
├── CDK2_HUMAN/
└── ...
```

---

## Kincore label analysis (single systems)

The "mother dataset" analysis: top-K success, number-of-correct distribution, and diversity.

### Run order

1. **comprehensive_method_analysis.py** – compute top-K success and per-file correctness
2. **extract_correct_predictions.py** – extract all-correct predictions and diversity subset (≥10 correct per method)
3. **analyze_correct_prediction_diversity.py** – diversity metrics on the subset
4. **plot_results.py** – main plots (success@K, num correct, diversity)
5. **plot_num_correct_distribution.py** – diversity-subset num-correct distribution

### Commands (from workspace root)

```bash
KINCORE=analysis/run_output/kincore_label
KINCONFBENCH=kinase_all_data/kinconfbench

# 1. Comprehensive analysis
python KinConfBench/analysis/kincore_label_analysis/comprehensive_method_analysis.py \
    $KINCONFBENCH -o $KINCORE/comprehensive_results

# 2. Extract correct predictions + diversity subset
python KinConfBench/analysis/kincore_label_analysis/extract_correct_predictions.py \
    $KINCONFBENCH -o $KINCORE/all_correct_analysis_csvs \
    -d $KINCORE/csvs_for_diversity_analysis -r $KINCORE/interesting_cases_report.txt

# 3. Diversity analysis
python KinConfBench/analysis/kincore_label_analysis/analyze_correct_prediction_diversity.py \
    -i $KINCORE/csvs_for_diversity_analysis -o $KINCORE/diversity_results

# 4. Main plots
python KinConfBench/analysis/kincore_label_analysis/plot_results.py \
    --accuracy $KINCORE/comprehensive_results_statistics.json \
    --accuracy-pkl $KINCORE/comprehensive_results_data.pkl \
    --diversity $KINCORE/diversity_results_diversity_summary.json \
    -o analysis/run_output/figures/results

# 5. Diversity subset num-correct distribution
python KinConfBench/analysis/kincore_label_analysis/plot_num_correct_distribution.py \
    $KINCORE/comprehensive_results_data.pkl -o analysis/run_output/figures/results_diversity \
    --subset-csv-dir $KINCORE/csvs_for_diversity_analysis
```

**`plot_results.py` options:** `--methods boltz,chai,protenix[,rfaa]` selects/overlays methods (colored in list order, see [Color scheme](#color-scheme)); `--diversity-csv-dir <dir>` adds the raw-diversity figure family; `--geometric-base-dir <root>` (optionally with `--geometric-success-only` / `--geometric-restrict-dataset`) adds the ligand-RMSD-merged `results_success_vs_k_with_ligand_rmsd.*` figure. The same flags apply to the successful-samples invocation below.

### Output figures

Always produced by `plot_results.py` (+ `plot_num_correct_distribution.py`), each as `.png` / `.pdf` / `.svg`:

- `results_success_vs_k.png`
- `results_num_correct_distribution.png`
- `results_diversity_num_correct_distribution.png`
- `results_average_diversity.png`
- `results_angular_diversity_detailed.png`
- `results_distance_diversity_detailed.png`
- `results_topk_accuracy.png`
- `results_first_correct_rank_cdf.png`

**On-demand figures** (only when the relevant flag is passed to `plot_results.py`):

- with `--geometric-base-dir`: `results_success_vs_k_with_ligand_rmsd.*` (Success@K curve + per-method ligand-RMSD boxplots; plus the legacy `results_success_vs_k_ligand_rmsd_stacked.*`).
- with `--diversity-csv-dir`: the raw-diversity family `results_raw_diversity_boxplot_3x4.*`, `results_raw_diversity_mean_std_bars_3x4.*`, `results_raw_diversity_bootstrap_bars_3x4.*` (each also emitted in a `*_with_reference.*` variant overlaying the KinConfBench reference), and `results_internal_std_boxplot_3x4.*`.

---

## Pair analysis (apo/holo seen/unseen)

**Generalization figure:** two-panel (**Seen** | **Unseen**) stacked bars of Holo/Apo outcome categories — **Holo State / Dual Match / Apo Drift / None** — with **method-specific deposition cutoffs**, in **pair_holo_apo_method_specific_analysis.py** (writes `figures/pair_holo_apo_categories_method_specific.png`).

**Prerequisite:** `holo_apo_results` with per-gene `holo_apo_pairs_by_labels.csv` (from pair extraction, e.g. `find_holo_apo_pairs_by_labels.py`).

### Run order

1. **create_pre_training_results.py** – filter to pre-cutoff and old ligands
2. **create_post_training_results.py** – filter to post-cutoff new ligands
3. **analyze_cofolding_results.py** – run on pre-training data
4. **analyze_cofolding_results.py** – run on post-training data
5. **analyze_method_performance.py** – once per pre and once per post (`method_performance_analysis_pre_training` / `_post_training`)
6. **pair_holo_apo_method_specific_analysis.py** – method-specific Seen/Unseen + stacked bar plot (pass `--pre_csv` / `--post_csv` or merged `--all_results`, plus `--pre_summaries` / `--post_summaries`)

### Commands

```bash
KINCONFBENCH=kinase_all_data/kinconfbench
HOLO_APO=analysis/pair_analysis_label/holo_apo_results
PRE=analysis/run_output/cofolding_results_pre_training
POST=analysis/run_output/cofolding_results_post_training

# 1. Pre-training subset
python KinConfBench/analysis/pair_analysis/create_pre_training_results.py \
    --source $KINCONFBENCH --output $PRE

# 2. Post-training subset
python KinConfBench/analysis/pair_analysis/create_post_training_results.py \
    --source $KINCONFBENCH --output $POST

# 3. Analyze cofolding (pre-training)
python KinConfBench/analysis/pair_analysis/analyze_cofolding_results.py \
    --holo_apo_results $HOLO_APO --cofolding_results_dir $PRE

# 4. Analyze cofolding (post-training)
python KinConfBench/analysis/pair_analysis/analyze_cofolding_results.py \
    --holo_apo_results $HOLO_APO --cofolding_results_dir $POST

# 5. Method performance (pre and post separately)
python KinConfBench/analysis/pair_analysis/analyze_method_performance.py \
    --results_dir $HOLO_APO --cofolding_results_dir $PRE

python KinConfBench/analysis/pair_analysis/analyze_method_performance.py \
    --results_dir $HOLO_APO --cofolding_results_dir $POST

# 6. Generalization plot (method-specific cutoffs)
python KinConfBench/analysis/pair_analysis/pair_holo_apo_method_specific_analysis.py \
    --pre_summaries $PRE --post_summaries $POST \
    --pre_csv $HOLO_APO/method_performance_analysis_pre_training/all_results_combined.csv \
    --post_csv $HOLO_APO/method_performance_analysis_post_training/all_results_combined.csv \
    --output_dir analysis/run_output/figures
```

### Output figures and tables

- `figures/pair_holo_apo_categories_method_specific.png` (and `.svg`)
- `all_results_combined.csv` — same holo$\rightarrow$apo rows as the merged pre/post inputs, but **`split` is reassigned per method** (each method’s deposition cutoff), not the coarse pre=Seen / post=Unseen merge alone.
- `pair_holo_apo_category_counts_method_specific.csv` — counts behind the stacked bars.

---

## Color scheme

Method figures get their colors from `method_colors()` in [`plot_style.py`](plot_style.py), which assigns colors **by index** from seaborn's `"deep"` palette and falls back to matplotlib's **tab10** cycle when seaborn is unavailable (the case in these envs). There are **no fixed hex codes** — colors follow the method list order:

| Method | tab10 index | Color |
|--------|-------------|-------|
| Boltz-2 | C0 | blue |
| Chai-1 | C1 | orange |
| Protenix | C2 | green |
| RoseTTAFold-AA (`rfaa`) | C3 | red |

Because the mapping is positional, adding a 4th method (`rfaa`) to any `--methods` list automatically gives it the next palette color — no config needed.

**Pair (apo/holo) categories** in `pair_holo_apo_method_specific_analysis.py` use tab10 hues directly (`_CATEGORY_STACK`): **Holo State** = blue (tab10[0]), **Dual Match** = blue/orange blend, **Apo Drift** = orange (tab10[1]), **None** = gray (tab10[7]). The figure has two panels, **Seen** and **Unseen**, split by each method's deposition cutoff.

---

## End-to-end pipeline (Boltz, Chai, Protenix; optional RFAA overlay)

The label and pair stages share the same **three-way geometric intersection**: each stem keeps only predictions passing the usual geometric thresholds, only stems where all three models have ≥1 successful prediction are exported, and output CSVs keep cofolding rows for **Boltz, Chai, and Protenix only**.

**Adding a 4th method (RoseTTAFold-AA).** `run_pipeline.py` threads three method lists so a 4th method overlays the paper's figures without changing the system set:

- `--methods boltz,chai,protenix,rfaa` — methods to **report** in the KinCoRe / diversity / plot stages.
- `--required-methods boltz,chai,protenix` — the **system gate** (defaults to `--methods`); keep it 3-way to preserve the paper's system set.
- `--keep-methods boltz,chai,protenix,rfaa` — methods whose prediction rows are **retained** in the successful subset (defaults to `--methods`).

Together these evaluate RFAA on exactly the same systems as the 3-way figures. `create_successful_samples_dataset.py` exposes the same `--required-methods` (gate) / `--keep-methods` (retained rows) split directly.

1. **Geometric analysis** on the full KinConfBench root — writes `geometric_analysis_csvs/*.csv` used when building the success set.
2. **`create_successful_samples_dataset.py`** (default `--required-methods boltz,chai,protenix`) — writes the successful-samples tree.
3. **KinCoRe branch** — `comprehensive_method_analysis.py`, `extract_correct_predictions.py`, and `analyze_correct_prediction_diversity.py` (defaults match the three methods) plus plotting scripts.
4. **Pair / apo-drift branch** — `create_pre_training_results.py` / `create_post_training_results.py` from that successful tree, then `analyze_cofolding_results.py` (pre and post), `analyze_method_performance.py` (once per pre and once per post), then `pair_holo_apo_method_specific_analysis.py` with matching `--pre_csv` / `--post_csv`.

Convenience driver (from repo root; adjust paths):

```bash
python KinConfBench/analysis/run_pipeline.py \
    --kinconfbench-root /path/to/full/kinconfbench \
    --holo-apo-results KinConfBench/analysis/pair_analysis_label/holo_apo_results

# Overlay RFAA on the paper's 3-way system set:
#   python KinConfBench/analysis/run_pipeline.py \
#       --kinconfbench-root /path/to/full/kinconfbench \
#       --holo-apo-results KinConfBench/analysis/pair_analysis_label/holo_apo_results \
#       --methods boltz,chai,protenix,rfaa --required-methods boltz,chai,protenix
```

Use `--skip-geometric` if `geometric_analysis_csvs` are already current on the source tree; use `--skip-pair` to stop after the diversity step.

Defaults write under `run_output/successful_samples_kinconfbench`, `run_output/kincore_label_successful`, `run_output/figures_successful`, and `run_output/cofolding_results_{pre,post}_training`. Override with `--successful-out`, `--kincore-out`, `--figures-out`, and `--pair-pre-dir` / `--pair-post-dir` for a self-contained bundle (for example under `outputs/`).

### Example bundle layout (optional `outputs/.../`)

Keep a single `--figures-out` so raw-diversity and main figures stay together (e.g. `figures_successful/raw_diversity_boxplots/`).

| Path | Contents |
|------|----------|
| `successful_samples_kinconfbench/` | Three-way geometric intersection tree; input to KinCoRe and pair steps. |
| `kincore_label_successful/` | `comprehensive_results_*`, `diversity_results_*`, `all_correct_analysis_csvs/`, `csvs_for_diversity_analysis/`, reports. |
| `figures_successful/` | Main plots (`results/`, `results_diversity/`, geometric correlation, optional `raw_diversity_boxplots/`, etc.). |
| `pair_holo_apo_*/` | Pair analysis CSVs and `figures/pair_holo_apo_categories_method_specific.*` when you point `--output_dir` there. |

Intermediate pair pre/post cofolding dirs default to `run_output/cofolding_results_pre_training` and `run_output/cofolding_results_post_training` unless overridden.

---

## Successful-samples pipeline (geometric success criteria)

To run the full pipeline restricted to **geometrically successful** samples (lDDT-Cα ≥ 0.7, lDDT-PLI ≥ 0.8, ligand_rmsd < 2 Å):

### Step 0: Create successful-samples dataset

```bash
python KinConfBench/analysis/create_successful_samples_dataset.py \
    --source kinase_all_data/kinconfbench \
    --output analysis/run_output/successful_samples_kinconfbench
```

### Single (KinCoRe) analysis on successful samples

```bash
KINCORE=analysis/run_output/kincore_label_successful
SUCCESSFUL=analysis/run_output/successful_samples_kinconfbench
FIGURES=analysis/run_output/figures_successful

python KinConfBench/analysis/kincore_label_analysis/comprehensive_method_analysis.py \
    $SUCCESSFUL -o $KINCORE/comprehensive_results

python KinConfBench/analysis/kincore_label_analysis/extract_correct_predictions.py \
    $SUCCESSFUL -o $KINCORE/all_correct_analysis_csvs \
    -d $KINCORE/csvs_for_diversity_analysis -r $KINCORE/interesting_cases_report.txt

python KinConfBench/analysis/kincore_label_analysis/analyze_correct_prediction_diversity.py \
    -i $KINCORE/csvs_for_diversity_analysis -o $KINCORE/diversity_results

python KinConfBench/analysis/kincore_label_analysis/plot_results.py \
    --accuracy $KINCORE/comprehensive_results_statistics.json \
    --accuracy-pkl $KINCORE/comprehensive_results_data.pkl \
    --diversity $KINCORE/diversity_results_diversity_summary.json \
    -o $FIGURES/results

python KinConfBench/analysis/kincore_label_analysis/plot_num_correct_distribution.py \
    $KINCORE/comprehensive_results_data.pkl -o $FIGURES/results_diversity \
    --subset-csv-dir $KINCORE/csvs_for_diversity_analysis
```

### Pair analysis on successful samples

```bash
PRE=analysis/run_output/cofolding_results_pre_training_successful
POST=analysis/run_output/cofolding_results_post_training_successful
HOLO_APO=analysis/pair_analysis_label/holo_apo_results
FIGURES=analysis/run_output/figures_successful

python KinConfBench/analysis/pair_analysis/create_pre_training_results.py \
    --source analysis/run_output/successful_samples_kinconfbench --output $PRE

python KinConfBench/analysis/pair_analysis/create_post_training_results.py \
    --source analysis/run_output/successful_samples_kinconfbench --output $POST

python KinConfBench/analysis/pair_analysis/analyze_cofolding_results.py \
    --holo_apo_results $HOLO_APO --cofolding_results_dir $PRE

python KinConfBench/analysis/pair_analysis/analyze_cofolding_results.py \
    --holo_apo_results $HOLO_APO --cofolding_results_dir $POST

python KinConfBench/analysis/pair_analysis/analyze_method_performance.py \
    --results_dir $HOLO_APO --cofolding_results_dir $PRE

python KinConfBench/analysis/pair_analysis/analyze_method_performance.py \
    --results_dir $HOLO_APO --cofolding_results_dir $POST

python KinConfBench/analysis/pair_analysis/pair_holo_apo_method_specific_analysis.py \
    --pre_summaries $PRE --post_summaries $POST \
    --pre_csv $HOLO_APO/method_performance_analysis_pre_training/all_results_combined.csv \
    --post_csv $HOLO_APO/method_performance_analysis_post_training/all_results_combined.csv \
    --output_dir $FIGURES \
    --methods boltz,chai,protenix
```

Add `rfaa` to `--methods` (e.g. `boltz,chai,protenix,rfaa`) to include RoseTTAFold-AA in the pair figure; the plot order puts the primary three first, then extras.

### Geometric correlation (successful-only)

```bash
python KinConfBench/analysis/kincore_label_analysis/geometric_correctness_correlation.py \
    --base-dir kinase_all_data/kinconfbench \
    -o analysis/run_output/figures_successful/kincore_geometric_correlation \
    --success-only
```

Emits, under the `-o` prefix: `_merged.csv`, per-metric correct-vs-incorrect boxplots (`_lddt_ca`, `_lddt_pli`, `_ligand_rmsd`, `_protein_rmsd`), stacked `_four_panel` / `_three_panel` grids, and `_overall_histograms.*`. Accepts `--methods` to match the reported method set.

---

## Folder layout

```
KinConfBench/analysis/
├── README.md
├── run_pipeline.py                      # Geometric → successful subset → KinCoRe → pair (3-way; +rfaa overlay via --methods)
├── create_successful_samples_dataset.py # Geometric-success gate → successful-samples tree (--required-methods / --keep-methods)
├── plot_style.py                        # Shared palette (method_colors, tab10/deep by index) + display names
├── geometric_analysis/                  # Geometric metrics (lDDT, RMSD)
├── kincore_label_analysis/              # Single-system correctness and diversity
│   ├── comprehensive_method_analysis.py     # top-K success / per-file correctness (data)
│   ├── extract_correct_predictions.py        # all-correct CSVs + diversity subset
│   ├── analyze_correct_prediction_diversity.py
│   ├── plot_results.py                       # main figures (success@K, diversity, +optional families)
│   ├── plot_num_correct_distribution.py
│   ├── geometric_correctness_correlation.py  # geometry-vs-correctness boxplots/panels
│   ├── plot_raw_diversity_boxplots.py        # raw-diversity 3×4 boxplots (helper)
│   ├── plot_raw_diversity_bootstrap_bars.py  # raw-diversity bootstrap bars (helper)
│   ├── plot_success_vs_k_ligand_rmsd_stacked.py
│   ├── plot_common_samples_distribution.py   # common-subset helpers
│   ├── plot_common_label_correct_summary.py
│   ├── si_spatial_dihedral_distribution_table.py   # SI table
│   └── combine_kincore_geometric_correlation_panels.py
└── pair_analysis/                       # Apo/holo seen/unseen
    ├── analyze_cofolding_results.py
    ├── create_pre_training_results.py
    ├── create_post_training_results.py
    ├── analyze_method_performance.py
    └── pair_holo_apo_method_specific_analysis.py
```

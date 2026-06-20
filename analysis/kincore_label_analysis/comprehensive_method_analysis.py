#!/usr/bin/env python3
"""
Comprehensive analysis across all directories with bootstrap confidence intervals.

Aggregates results from multiple base directories and computes statistical
confidence levels using bootstrap resampling.

Usage:
    python comprehensive_method_analysis.py /path/to/kinconfbench/
    python comprehensive_method_analysis.py /path/to/kinconfbench/ -o output/prefix
"""

import csv
import json
import pickle
import os
import glob
import argparse
import random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def matches_any_ground_truth(pred_row, ground_truth_rows, label_keys):
    """Check if prediction matches any ground truth row for specified labels."""
    for gt_row in ground_truth_rows:
        if all(pred_row.get(key) == gt_row.get(key) for key in label_keys):
            return True
    return False


DEFAULT_METHODS = ('boltz', 'chai', 'protenix')


def calculate_statistics_for_csv(csv_path, methods=DEFAULT_METHODS):
    """Calculate all-labels correctness for each method+conf_rank."""

    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        ground_truth = [r for r in rows if len(r['structure_id']) == 4 and
                        r['structure_id'].replace('_','').replace('-','').isalnum()]

        if not ground_truth:
            return None

        all_label_keys = [
            'activity_label', 'spatial_label', 'dihedral_label',
            'chelix_label', 'saltbr_label', 'actloop_nt_label',
            'actloop_ct_label', 'spine_label'
        ]

        results_with_conf = {}

        for method in methods:
            prefix = f"{method}_"
            method_rows = [r for r in rows if r['structure_id'].startswith(prefix)]

            if not method_rows:
                continue

            conf_results = {}

            for r in method_rows:
                conf_rank = extract_conf_rank(r['structure_id'])

                if conf_rank is not None:
                    all_correct = matches_any_ground_truth(r, ground_truth, all_label_keys)
                    conf_results[conf_rank] = int(all_correct)

            results_with_conf[method] = conf_results

        return results_with_conf

    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        return None


def extract_conf_rank(structure_id):
    """Extract confidence rank from structure_id (e.g., 'boltz_conf0' -> 0)."""
    import re
    match = re.search(r'_conf(\d+)', structure_id)
    if match:
        return int(match.group(1))
    return None


def calculate_topk_system_success_rate(per_file_stats_with_conf, methods=DEFAULT_METHODS,
                                       n_bootstrap=1000):
    """Calculate top-K system success rates for K=1,5,10,20."""

    topk_configs = {
        'top1': 0,
        'top5': 4,
        'top10': 9,
        'top20': 19
    }

    topk_success = {k: {} for k in topk_configs.keys()}

    for method in methods:
        for topk_name, max_conf in topk_configs.items():
            file_stats = per_file_stats_with_conf[method]

            system_binary = []
            for file_name, conf_dict in file_stats.items():
                has_correct = False
                for conf_rank, is_correct in conf_dict.items():
                    if conf_rank <= max_conf and is_correct:
                        has_correct = True
                        break
                system_binary.append(1 if has_correct else 0)

            total_systems = len(system_binary)
            systems_with_success = sum(system_binary)
            success_rate = 100 * systems_with_success / total_systems if total_systems > 0 else 0

            if total_systems > 0:
                bootstrap_rates = []
                for _ in range(n_bootstrap):
                    sample = random.choices(system_binary, k=total_systems)
                    bootstrap_rates.append(100 * sum(sample) / len(sample))

                bootstrap_rates.sort()
                alpha = 0.025
                lower_idx = int(alpha * n_bootstrap)
                upper_idx = int((1 - alpha) * n_bootstrap)
                ci_lower = bootstrap_rates[lower_idx]
                ci_upper = bootstrap_rates[upper_idx]
            else:
                ci_lower = 0
                ci_upper = 0

            topk_success[topk_name][method] = {
                'total_systems': total_systems,
                'systems_with_success': systems_with_success,
                'success_rate': success_rate,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'ci_width': ci_upper - ci_lower
            }

    return topk_success


def print_topk_system_success_summary(topk_success, methods=DEFAULT_METHODS):
    """Print top-K system-level success rate summary."""

    print("=" * 90)
    print("TOP-K SYSTEM-LEVEL SUCCESS RATE")
    print("=" * 90)
    print()

    for topk_name in ['top1', 'top5', 'top10', 'top20']:
        print(f"{topk_name.upper()} (conf0" +
              (f"-{topk_name[3:]}" if topk_name != 'top1' else "") + "):")
        print("-" * 90)
        for method in methods:
            stats = topk_success[topk_name][method]
            print(f"{method.upper():>10s}: {stats['systems_with_success']:4d}/{stats['total_systems']:4d} "
                  f"({stats['success_rate']:5.1f}% +/- {stats['ci_width']/2:4.1f}%) "
                  f"[{stats['ci_lower']:5.1f}%, {stats['ci_upper']:5.1f}%]")
        print()

    print()


def comprehensive_analysis(base_dirs, output_prefix=None, methods=DEFAULT_METHODS):
    """Run comprehensive analysis across all directories."""

    methods = tuple(methods)
    all_csv_files = []
    for base_dir in base_dirs:
        pattern = os.path.join(base_dir, '*/analysis_csvs/*_analysis.csv')
        csv_files = glob.glob(pattern)
        all_csv_files.extend(csv_files)
        print(f"Found {len(csv_files)} CSV files in {base_dir}")

    print(f"\nTotal CSV files to analyze: {len(all_csv_files)}")
    print()

    per_file_stats_with_conf = {method: {} for method in methods}

    file_methods = {}
    processed = 0
    skipped = 0
    skipped_files = []
    batch_size = 100

    for batch_start in range(0, len(all_csv_files), batch_size):
        batch_end = min(batch_start + batch_size, len(all_csv_files))
        batch = all_csv_files[batch_start:batch_end]
        batch_num = batch_start//batch_size + 1
        total_batches = (len(all_csv_files) + batch_size - 1)//batch_size

        print(f"\nProcessing batch {batch_num}/{total_batches} "
              f"(files {batch_start+1}-{batch_end}/{len(all_csv_files)})", flush=True)

        for i, csv_path in enumerate(tqdm(batch, desc=f"  Batch {batch_num}")):
            file_name = Path(csv_path).stem.replace('_analysis', '')

            results_with_conf = calculate_statistics_for_csv(csv_path, methods=methods)

            if results_with_conf is None:
                skipped += 1
                skipped_files.append(file_name)
                continue

            file_methods[file_name] = set(results_with_conf.keys())

            for method, conf_results in results_with_conf.items():
                per_file_stats_with_conf[method][file_name] = conf_results

            processed += 1

            if (i + 1) % 10 == 0 or (i + 1) == len(batch):
                print(f"  Processed {i+1}/{len(batch)} files in batch {batch_num} "
                      f"({processed} total)", flush=True)

    print(f"\nSuccessfully processed: {processed}")
    print(f"Skipped (no ground truth): {skipped}")
    print()

    all_methods = frozenset(methods)
    files_missing_methods = {}
    complete_files = []

    for file_name, present in file_methods.items():
        missing = all_methods - present
        if missing:
            files_missing_methods[file_name] = missing
        else:
            complete_files.append(file_name)

    print(f"\n{'='*90}")
    print(f"Files with all {len(methods)} methods: {len(complete_files)}")
    print(f"Files missing one or more methods: {len(files_missing_methods)}")
    print(f"{'='*90}")

    if files_missing_methods:
        print(f"\nExcluding {len(files_missing_methods)} files from analysis:")

        by_missing = {}
        for file_name, missing in files_missing_methods.items():
            missing_str = ', '.join(sorted(missing))
            if missing_str not in by_missing:
                by_missing[missing_str] = []
            by_missing[missing_str].append(file_name)

        for missing_str, files in sorted(by_missing.items()):
            print(f"  Missing {missing_str}: {len(files)} files")
        print()

    filtered_stats = {
        method: {fn: data for fn, data in per_file_stats_with_conf[method].items()
                 if fn in complete_files}
        for method in methods
    }

    print(f"Proceeding with {len(complete_files)} files that have all {len(methods)} methods\n")

    print("Calculating top-K system-level success rates...")
    topk_success = calculate_topk_system_success_rate(filtered_stats, methods=methods)

    print_topk_system_success_summary(topk_success, methods=methods)

    if output_prefix:
        Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)

        pickle_file = f"{output_prefix}_data.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump({
                'per_file_stats_with_conf': per_file_stats_with_conf,
                'topk_success': topk_success
            }, f)
        print(f"Data saved to {pickle_file}")

        json_file = f"{output_prefix}_statistics.json"
        with open(json_file, 'w') as f:
            json.dump(topk_success, f, indent=2)
        print(f"Statistics saved to {json_file}")

        txt_file = f"{output_prefix}_report.txt"
        generate_topk_report(txt_file, topk_success, processed, skipped, skipped_files,
                           files_missing_methods, len(complete_files), methods=methods)
        print(f"Comprehensive report saved to {txt_file}")

    return topk_success, per_file_stats_with_conf


def generate_topk_report(output_file, topk_success, processed, skipped, skipped_files,
                        files_missing_methods, complete_files_count, methods=DEFAULT_METHODS):
    """Generate report with top-K system success rates."""

    with open(output_file, 'w') as f:
        f.write("=" * 90 + "\n")
        f.write("TOP-K SYSTEM-LEVEL SUCCESS RATE ANALYSIS\n")
        f.write("=" * 90 + "\n\n")

        f.write(f"Total files processed: {processed}\n")
        f.write(f"Total files skipped (no ground truth): {skipped}\n")
        f.write(f"Total files excluded (missing methods): {len(files_missing_methods)}\n")
        f.write(
            f"Total files included in analysis (all {len(methods)} methods present): "
            f"{complete_files_count}\n\n"
        )

        f.write("\n\n")
        f.write("=" * 90 + "\n")
        f.write("TOP-K SYSTEM-LEVEL SUCCESS RATE\n")
        f.write("=" * 90 + "\n\n")

        f.write("System success rates considering only top-K predictions per system:\n")
        f.write("(K predictions ranked by confidence: top-1=conf0, top-5=conf0-4, etc.)\n\n")

        for topk_name in ['top1', 'top5', 'top10', 'top20']:
            topk_label = topk_name.upper()
            if topk_name == 'top1':
                conf_range = "conf0"
            else:
                conf_range = f"conf0-{int(topk_name[3:])-1}"

            f.write(f"{topk_label} ({conf_range}):\n")
            f.write("-" * 90 + "\n")

            topk_rankings = []
            for method in methods:
                stats = topk_success[topk_name][method]
                topk_rankings.append((
                    method,
                    stats['systems_with_success'],
                    stats['total_systems'],
                    stats['success_rate'],
                    stats['ci_lower'],
                    stats['ci_upper'],
                    stats['ci_width']
                ))

            topk_rankings_sorted = sorted(topk_rankings, key=lambda x: x[3], reverse=True)

            for i, (method, success, total, rate, ci_low, ci_high, ci_width) in enumerate(topk_rankings_sorted, 1):
                f.write(f"  {i}. {method.upper():10s}: {success:4d}/{total:4d} "
                       f"({rate:5.1f}% +/- {ci_width/2:4.1f}%) "
                       f"[{ci_low:5.1f}%, {ci_high:5.1f}%]\n")

            f.write("\n")

        f.write("Key observations:\n")
        f.write("- Top-K metrics show how well methods perform when only considering\n")
        f.write("  their K most confident predictions per system\n")
        f.write("- Comparing top-1 vs top-20 shows diversity: higher top-20 suggests\n")
        f.write("  the method generates diverse predictions\n")
        f.write("- Top-1 accuracy shows how well the most confident prediction performs\n\n")

        f.write("Improvement from top-1 to top-20:\n")
        for method in methods:
            top1_rate = topk_success['top1'][method]['success_rate']
            top20_rate = topk_success['top20'][method]['success_rate']
            improvement = top20_rate - top1_rate
            f.write(f"  {method.upper():10s}: {top1_rate:5.1f}% -> {top20_rate:5.1f}% "
                   f"(+{improvement:5.1f} percentage points)\n")

        if files_missing_methods:
            f.write("\n\n")
            f.write("=" * 90 + "\n")
            f.write("FILES EXCLUDED FROM ANALYSIS (Missing one or more methods)\n")
            f.write("=" * 90 + "\n\n")

            f.write(f"Total excluded: {len(files_missing_methods)} files\n")
            f.write(f"These files were not included in the top-K statistics above.\n\n")

            by_missing = {}
            for file_name, missing in files_missing_methods.items():
                missing_str = ', '.join(sorted(missing))
                if missing_str not in by_missing:
                    by_missing[missing_str] = []
                by_missing[missing_str].append(file_name)

            for missing_str, files in sorted(by_missing.items()):
                f.write(f"\nMissing {missing_str}: {len(files)} files\n")
                f.write("-" * 90 + "\n")
                for fn in sorted(files):
                    f.write(f"  {fn}\n")

        f.write("\n\n")
        f.write("=" * 90 + "\n")
        f.write("FILES WITHOUT GROUND TRUTH\n")
        f.write("=" * 90 + "\n\n")

        f.write(f"Total: {len(skipped_files)} files\n\n")

        if skipped_files:
            skipped_files_sorted = sorted(skipped_files)

            f.write("List of files (first 100):\n")
            for i, file_name in enumerate(skipped_files_sorted[:100], 1):
                f.write(f"  {i:3d}. {file_name}\n")

            if len(skipped_files) > 100:
                f.write(f"\n  ... and {len(skipped_files) - 100} more files\n")

            f.write(f"\n\nComplete list of all {len(skipped_files)} files:\n")
            for file_name in skipped_files_sorted:
                f.write(f"{file_name}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive analysis across all directories with bootstrap CI'
    )
    parser.add_argument(
        'base_dirs',
        nargs='+',
        help='Base directories containing gene folders with analysis_csvs/ subdirectories'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file prefix for results',
        default='comprehensive_results'
    )
    parser.add_argument(
        '--methods',
        type=str,
        default=','.join(DEFAULT_METHODS),
        help='Comma-separated methods that must be present per system (default: boltz,chai,protenix)',
    )

    args = parser.parse_args()
    mlist = tuple(x.strip().lower() for x in args.methods.split(',') if x.strip())
    if not mlist:
        raise SystemExit("--methods must list at least one method")

    comprehensive_analysis(args.base_dirs, args.output, methods=mlist)


if __name__ == '__main__':
    main()

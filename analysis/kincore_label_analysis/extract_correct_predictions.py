#!/usr/bin/env python3
"""
Extract all-correct predictions from analysis CSVs and identify interesting cases.

Creates folders with CSVs containing only predictions that got all labels correct,
along with ground truth rows. Also identifies cases where methods perform well or poorly.
Only includes files where every requested method (default: boltz, chai, protenix) is present.
Additionally filters files for diversity analysis (>=10 all-correct predictions per method).

Usage:
    python extract_correct_predictions.py /path/to/kinconfbench/
    python extract_correct_predictions.py /path/to/kinconfbench/ -o all_correct_csvs -d diversity_csvs
"""

import csv
import os
import glob
import json
import argparse
import shutil
from pathlib import Path
from collections import defaultdict

from KinConfBench.analysis.plot_style import PRIMARY_METHODS

DEFAULT_METHODS = PRIMARY_METHODS


def matches_any_ground_truth(pred_row, ground_truth_rows, label_keys):
    """Check if prediction matches any ground truth row for specified labels."""
    for gt_row in ground_truth_rows:
        if all(pred_row.get(key) == gt_row.get(key) for key in label_keys):
            return True
    return False


def process_csv_for_correct_predictions(csv_path, methods=DEFAULT_METHODS):
    """
    Extract all-correct predictions from a CSV.

    Returns result dict without saving (saving is done separately based on filtering).
    """

    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
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

        method_correct_counts = {}
        method_total_counts = {}
        methods_present = set()
        all_correct_rows = []

        for method in methods:
            prefix = f"{method}_"
            method_rows = [r for r in rows if r['structure_id'].startswith(prefix)]

            if not method_rows:
                method_correct_counts[method] = 0
                method_total_counts[method] = 0
                continue

            methods_present.add(method)

            correct_rows = [r for r in method_rows
                           if matches_any_ground_truth(r, ground_truth, all_label_keys)]

            method_correct_counts[method] = len(correct_rows)
            method_total_counts[method] = len(method_rows)
            all_correct_rows.extend(correct_rows)

        return {
            'file_name': Path(csv_path).stem.replace('_analysis', ''),
            'csv_path': csv_path,
            'fieldnames': fieldnames,
            'ground_truth': ground_truth,
            'all_correct_rows': all_correct_rows,
            'ground_truth_count': len(ground_truth),
            'method_counts': method_correct_counts,
            'method_totals': method_total_counts,
            'methods_present': methods_present,
            'total_correct': sum(method_correct_counts.values())
        }

    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        return None


def save_correct_predictions_csv(result, output_dir):
    """Save all-correct predictions CSV to output directory."""

    file_name = Path(result['csv_path']).name
    output_path = os.path.join(output_dir, file_name)

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=result['fieldnames'])
        writer.writeheader()

        for gt_row in result['ground_truth']:
            writer.writerow(gt_row)

        for correct_row in result['all_correct_rows']:
            writer.writerow(correct_row)


def _case_row(file_name: str, pcts: dict, counts: dict, totals: dict, methods: tuple[str, ...]) -> dict:
    row: dict = {"file": file_name}
    for m in methods:
        row[m] = pcts[m]
        row[f"{m}_count"] = f"{counts.get(m, 0)}/{totals.get(m, 0)}"
    return row


def identify_interesting_cases(all_results, method_totals, methods=DEFAULT_METHODS):
    """Identify cases where methods perform well or poorly."""

    all_methods_well: list[dict] = []
    all_methods_fail: list[dict] = []
    all_methods_zero: list[dict] = []

    methods = tuple(methods)

    for result in all_results:
        if result is None:
            continue

        file_name = result["file_name"]
        counts = result["method_counts"]
        totals = result["method_totals"]

        if any(m not in counts for m in methods):
            continue

        pcts = {}
        for m in methods:
            pcts[m] = (
                counts.get(m, 0) / totals.get(m, 1) * 100 if totals.get(m, 0) > 0 else 0
            )

        case_data = _case_row(file_name, pcts, counts, totals, methods)

        method_pcts = [pcts[m] for m in methods]
        if all(pct > 50 for pct in method_pcts):
            all_methods_well.append(case_data.copy())

        if all(pcts[m] < 10 for m in methods):
            all_methods_fail.append(case_data.copy())

        if all(counts.get(m, 0) == 0 for m in methods):
            all_methods_zero.append(case_data.copy())

    return {
        "all_methods_well": all_methods_well,
        "all_methods_fail": all_methods_fail,
        "all_methods_zero": all_methods_zero,
    }


def main_analysis(base_dirs, output_dir, report_file, diversity_dir=None, diversity_min=10,
                  methods=DEFAULT_METHODS):
    """Main analysis function."""

    os.makedirs(output_dir, exist_ok=True)
    if diversity_dir:
        os.makedirs(diversity_dir, exist_ok=True)

    all_csv_files = []
    for base_dir in base_dirs:
        pattern = os.path.join(base_dir, '*/analysis_csvs/*_analysis.csv')
        csv_files = glob.glob(pattern)
        all_csv_files.extend(csv_files)

    print(f"Found {len(all_csv_files)} CSV files to process")
    print()

    methods = tuple(methods)
    all_results = []
    processed = 0

    for csv_path in all_csv_files:
        result = process_csv_for_correct_predictions(csv_path, methods=methods)
        all_results.append(result)

        processed += 1
        if processed % 100 == 0:
            print(f"Processed {processed} files...")

    print()
    print(f"Total files processed: {processed}")
    print()

    all_methods = frozenset(methods)
    complete_results = []
    files_missing_methods = {}

    for result in all_results:
        if result is None:
            continue

        missing = all_methods - result['methods_present']
        if missing:
            files_missing_methods[result['file_name']] = missing
        else:
            complete_results.append(result)

    print(f"{'='*90}")
    print(f"Files with all {len(methods)} methods: {len(complete_results)}")
    print(f"Files missing one or more methods: {len(files_missing_methods)}")
    print(f"{'='*90}")
    print()

    if files_missing_methods:
        print(f"Excluding {len(files_missing_methods)} files from extraction:")

        by_missing = {}
        for file_name, missing in files_missing_methods.items():
            missing_str = ', '.join(sorted(missing))
            if missing_str not in by_missing:
                by_missing[missing_str] = []
            by_missing[missing_str].append(file_name)

        for missing_str, files in sorted(by_missing.items()):
            print(f"  Missing {missing_str}: {len(files)} files")
        print()

    saved = 0
    diversity_candidates = []

    for result in complete_results:
        if result['total_correct'] > 0:
            save_correct_predictions_csv(result, output_dir)
            saved += 1

            if all(result['method_counts'].get(method, 0) >= diversity_min
                   for method in methods):
                diversity_candidates.append(result)

    print(f"Saved {saved} files with all-correct predictions (all {len(methods)} methods present)")
    print()

    if diversity_dir and diversity_candidates:
        print(f"Filtering for diversity analysis (>={diversity_min} all-correct per method)...")
        for result in diversity_candidates:
            file_name = Path(result['csv_path']).name
            src = os.path.join(output_dir, file_name)
            dst = os.path.join(diversity_dir, file_name)
            shutil.copy2(src, dst)

        print(f"Copied {len(diversity_candidates)} files to diversity analysis directory")
        print()

        diversity_summary = os.path.join(diversity_dir, "diversity_selection_summary.txt")
        generate_diversity_summary(diversity_summary, diversity_candidates, diversity_min,
                                   methods=methods)
        print(f"Diversity summary saved to {diversity_summary}")
        print()

    print("Identifying interesting cases...")
    interesting_cases = identify_interesting_cases(complete_results, None, methods=methods)

    if output_dir:
        json_file = os.path.join(os.path.dirname(report_file), "interesting_cases.json")
        Path(json_file).parent.mkdir(parents=True, exist_ok=True)
        with open(json_file, 'w') as f:
            json.dump(interesting_cases, f, indent=2)
        print(f"Interesting cases saved to {json_file}")

    Path(report_file).parent.mkdir(parents=True, exist_ok=True)
    generate_report(report_file, complete_results, interesting_cases, processed, saved,
                   files_missing_methods, len(diversity_candidates) if diversity_dir else 0,
                   methods=methods)
    print(f"Report saved to {report_file}")


def generate_diversity_summary(summary_file, diversity_candidates, diversity_min,
                               methods=DEFAULT_METHODS):
    """Generate diversity analysis summary."""

    methods = tuple(methods)
    file_stats = []
    for result in diversity_candidates:
        counts = result['method_counts']
        row = {'file': result['file_name'], 'total': result['total_correct']}
        for m in methods:
            row[m] = counts.get(m, 0)
        file_stats.append(row)

    with open(summary_file, 'w') as f:
        f.write("=" * 90 + "\n")
        f.write("DIVERSITY ANALYSIS - FILE SELECTION SUMMARY\n")
        f.write("=" * 90 + "\n\n")

        f.write(
            f"Selection criteria: Methods {', '.join(methods)} each have >={diversity_min} "
            "all-correct predictions\n"
        )
        f.write(f"Total files selected: {len(diversity_candidates)}\n\n")

        f.write("=" * 90 + "\n")
        f.write("SELECTED FILES\n")
        f.write("=" * 90 + "\n\n")

        file_stats_sorted = sorted(file_stats, key=lambda x: x['total'], reverse=True)

        hdr = f"{'File Name':60s}"
        for m in methods:
            hdr += f" {m.upper():>8s}"
        hdr += f" {'Total':>6s}\n"
        f.write(hdr)
        f.write("-" * 90 + "\n")

        for stat in file_stats_sorted:
            line = f"{stat['file']:60s}"
            for m in methods:
                line += f" {stat[m]:8d}"
            line += f" {stat['total']:6d}\n"
            f.write(line)

        f.write("\n\n")
        f.write("=" * 90 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("=" * 90 + "\n\n")

        total_by_method = {m: sum(s[m] for s in file_stats) for m in methods}

        f.write("Total all-correct predictions in selected files:\n")
        for method, count in total_by_method.items():
            f.write(f"  {method.upper():10s}: {count:5d}\n")

        f.write(f"\nTotal all-correct predictions: {sum(total_by_method.values())}\n")

        f.write("\n\nDistribution by method:\n")
        all_counts_by_method = {m: [s[m] for s in file_stats] for m in methods}

        for method, counts in all_counts_by_method.items():
            if counts:
                avg = sum(counts) / len(counts)
                min_val = min(counts)
                max_val = max(counts)
                f.write(f"  {method.upper():10s}: min={min_val:2d}, max={max_val:3d}, avg={avg:5.1f}\n")


def generate_report(report_file, all_results, interesting_cases, processed, saved,
                   files_missing_methods, diversity_count, methods=DEFAULT_METHODS):
    """Generate report with interesting cases."""

    methods = tuple(methods)
    with open(report_file, 'w') as f:
        f.write("=" * 90 + "\n")
        f.write("CORRECT PREDICTIONS EXTRACTION REPORT\n")
        f.write("=" * 90 + "\n\n")

        f.write(f"Total files processed: {processed}\n")
        f.write(f"Files excluded (missing methods): {len(files_missing_methods)}\n")
        f.write(f"Files with all {len(methods)} methods and all-correct predictions: {saved}\n")
        if diversity_count > 0:
            f.write(f"Files qualifying for diversity analysis: {diversity_count}\n")
        f.write("\n")

        f.write("=" * 90 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("=" * 90 + "\n\n")

        total_correct_by_method = defaultdict(int)
        files_with_correct = defaultdict(int)

        for result in all_results:
            if result is None:
                continue

            for method, count in result['method_counts'].items():
                total_correct_by_method[method] += count
                if count > 0:
                    files_with_correct[method] += 1

        f.write("Total all-correct predictions by method:\n")
        for method in methods:
            f.write(f"  {method.upper():10s}: {total_correct_by_method[method]:5d} predictions "
                   f"from {files_with_correct[method]:3d} files\n")

        f.write("\n\n")
        f.write("=" * 90 + "\n")
        f.write("INTERESTING CASES\n")
        f.write("=" * 90 + "\n\n")

        nmeth = len(methods)
        label_all = f"ALL {nmeth} methods"
        f.write(f"Cases where {label_all} perform well (>50% correct):\n")
        f.write(f"Total: {len(interesting_cases['all_methods_well'])} cases\n\n")

        if interesting_cases["all_methods_well"]:
            sorted_cases = sorted(
                interesting_cases["all_methods_well"],
                key=lambda x: sum(x[m] for m in methods) / nmeth,
                reverse=True,
            )

            f.write("Top 10 cases:\n")
            for i, case in enumerate(sorted_cases[:10], 1):
                avg = sum(case[m] for m in methods) / nmeth
                f.write(f"{i:2d}. {case['file']:50s} (avg: {avg:5.1f}%)\n")
                parts = []
                for m in methods:
                    parts.append(
                        f"{m.upper()}: {case[m]:5.1f}% ({case[f'{m}_count']:>7s})"
                    )
                f.write("    " + "  ".join(parts) + "\n")

        f.write(f"\n\nCases where {label_all} struggle (<10% correct):\n")
        f.write(f"Total: {len(interesting_cases['all_methods_fail'])} cases\n\n")

        if interesting_cases["all_methods_fail"]:
            sorted_cases = sorted(
                interesting_cases["all_methods_fail"],
                key=lambda x: sum(x[m] for m in methods) / nmeth,
            )

            f.write("First 10 cases (sorted by average performance):\n")
            for i, case in enumerate(sorted_cases[:10], 1):
                avg = sum(case[m] for m in methods) / nmeth
                f.write(f"{i:2d}. {case['file']:50s} (avg: {avg:4.1f}%)\n")
                parts = []
                for m in methods:
                    parts.append(
                        f"{m.upper()}: {case[m]:4.1f}% ({case[f'{m}_count']:>7s})"
                    )
                f.write("    " + "  ".join(parts) + "\n")

        f.write(f"\n\nCases where {label_all} have ZERO correct:\n")
        f.write(f"Total: {len(interesting_cases['all_methods_zero'])} cases\n\n")
        if interesting_cases["all_methods_zero"]:
            sorted_zero = sorted([c["file"] for c in interesting_cases["all_methods_zero"]])
            f.write("All systems (one per line):\n")
            for fn in sorted_zero:
                f.write(f"  {fn}\n")
            report_dir = os.path.dirname(report_file)
            zero_list_path = os.path.join(report_dir, "all_methods_zero_systems.txt")
            with open(zero_list_path, "w") as zf:
                zf.write(
                    f"# Systems where all {nmeth} methods have 0 correct (out of top-20)\n"
                )
                zf.write(f"# Total: {len(sorted_zero)}\n\n")
                for fn in sorted_zero:
                    zf.write(fn + "\n")
            print(f"Wrote full list to {zero_list_path}")

        if files_missing_methods:
            f.write("\n\n")
            f.write("=" * 90 + "\n")
            f.write("FILES EXCLUDED FROM EXTRACTION (Missing one or more methods)\n")
            f.write("=" * 90 + "\n\n")

            f.write(f"Total excluded: {len(files_missing_methods)} files\n")
            f.write(f"These files were not included in the extraction.\n\n")

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


def main():
    parser = argparse.ArgumentParser(
        description="Extract all-correct predictions and identify interesting cases. "
        "Only includes files where every listed method is present."
    )
    parser.add_argument(
        'base_dirs',
        nargs='+',
        help='Base directories containing gene folders with analysis_csvs/ subdirectories'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output directory for all-correct CSVs',
        default='all_correct_analysis_csvs'
    )
    parser.add_argument(
        '-r', '--report',
        help='Output report file',
        default='interesting_cases_report.txt'
    )
    parser.add_argument(
        '-d', '--diversity-dir',
        help='Output directory for diversity analysis CSVs (files with >=M all-correct per method)',
        default='csvs_for_diversity_analysis'
    )
    parser.add_argument(
        '-m', '--diversity-min',
        type=int,
        help='Minimum number of all-correct predictions per method for diversity analysis',
        default=10
    )
    parser.add_argument(
        '--no-diversity',
        action='store_true',
        help='Skip diversity filtering'
    )
    parser.add_argument(
        '--methods',
        type=str,
        default=','.join(DEFAULT_METHODS),
        help='Comma-separated methods that must be present per system (default: boltz,chai,protenix)',
    )

    args = parser.parse_args()

    diversity_dir = None if args.no_diversity else args.diversity_dir
    mlist = tuple(x.strip().lower() for x in args.methods.split(',') if x.strip())
    if not mlist:
        raise SystemExit("--methods must list at least one method")

    main_analysis(args.base_dirs, args.output, args.report, diversity_dir, args.diversity_min,
                  methods=mlist)


if __name__ == '__main__':
    main()

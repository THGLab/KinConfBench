#!/usr/bin/env python3
"""
Analyze structural diversity of all-correct predictions.

For files where all methods have >=10 all-correct predictions,
calculates numerical diversity metrics (distances and angles) to understand
conformational sampling among correct predictions.

Note: Categorical labels (activity, spatial, etc.) are excluded since all
predictions are all-correct and should match ground truth exactly.

Usage:
    python analyze_correct_prediction_diversity.py -i csvs_for_diversity_analysis/ -o diversity_results
"""

import csv
import json
import pickle
import os
import glob
import random
import argparse
from pathlib import Path
from collections import defaultdict
import math

DEFAULT_METHODS = ('boltz', 'chai', 'protenix')

DISTANCE_COLUMNS = [
    "lys_glu_distance",
    "saltbridge_distance",
    "spine_distance",
    "dfg_hrd_distance",
]

ANGLE_COLUMNS = [
    "asp_phi", "asp_psi", "asp_chi1", "asp_chi2",
    "phe_phi", "phe_psi", "phe_chi1", "phe_chi2"
]

PI_SYMMETRY_ANGLES = ["asp_chi2", "phe_chi2"]


def circular_variance(angles_deg, has_pi_symmetry=False):
    """Calculate circular variance for angles."""
    if len(angles_deg) == 0:
        return 0.0

    angles_rad = [math.radians(a) for a in angles_deg]

    if has_pi_symmetry:
        angles_rad = [2 * a for a in angles_rad]

    mean_cos = sum(math.cos(a) for a in angles_rad) / len(angles_rad)
    mean_sin = sum(math.sin(a) for a in angles_rad) / len(angles_rad)
    R = math.sqrt(mean_cos**2 + mean_sin**2)

    return 1 - R


def coefficient_of_variation(values):
    """Calculate coefficient of variation for distance metrics."""
    if len(values) <= 1:
        return 0.0

    mean_val = sum(values) / len(values)
    if mean_val == 0:
        return 0.0

    variance = sum((x - mean_val)**2 for x in values) / (len(values) - 1)
    std_val = math.sqrt(variance)

    return std_val / mean_val


def analyze_csv_diversity(csv_path, methods=DEFAULT_METHODS):
    """Analyze diversity of all-correct predictions in a single CSV."""

    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        ground_truth = [r for r in rows if len(r['structure_id']) == 4]

        method_data = {
            m: [r for r in rows if r['structure_id'].startswith(f"{m}_")]
            for m in methods
        }

        results = {}

        for method, method_rows in method_data.items():
            if not method_rows:
                continue

            diversity = {
                'n_predictions': len(method_rows),
                'distance_cv': {},
                'angular_variance': {}
            }

            for col in DISTANCE_COLUMNS:
                values = []
                for row in method_rows:
                    try:
                        val = float(row.get(col, ''))
                        values.append(val)
                    except (ValueError, TypeError):
                        pass

                if len(values) > 1:
                    diversity['distance_cv'][col] = coefficient_of_variation(values)

            for col in ANGLE_COLUMNS:
                values = []
                for row in method_rows:
                    try:
                        val = float(row.get(col, ''))
                        values.append(val)
                    except (ValueError, TypeError):
                        pass

                if len(values) > 1:
                    has_pi_symmetry = col in PI_SYMMETRY_ANGLES
                    diversity['angular_variance'][col] = circular_variance(values, has_pi_symmetry)

            results[method] = diversity

        return results

    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        return None


def aggregate_diversity_analysis(csv_dir, output_prefix, methods=DEFAULT_METHODS):
    """Aggregate diversity analysis across all CSVs."""

    methods = tuple(methods)
    csv_files = list(Path(csv_dir).glob("*.csv"))
    csv_files = [f for f in csv_files if not f.name.startswith('diversity_')]

    print(f"Analyzing diversity for {len(csv_files)} CSV files")
    print()

    all_results = []
    aggregated_diversity = {m: defaultdict(list) for m in methods}

    processed = 0

    for csv_file in csv_files:
        file_name = csv_file.stem.replace('_analysis', '')

        results = analyze_csv_diversity(csv_file, methods=methods)

        if results is None:
            continue

        file_result = {'file': file_name}
        for method, diversity in results.items():
            file_result[method] = diversity

            for metric_type in ['distance_cv', 'angular_variance']:
                for feature, value in diversity.get(metric_type, {}).items():
                    aggregated_diversity[method][f"{metric_type}_{feature}"].append(value)

        all_results.append(file_result)

        processed += 1
        if processed % 100 == 0:
            print(f"Processed {processed} files...")

    print()
    print(f"Successfully processed: {processed}")
    print()

    print_diversity_summary(aggregated_diversity, methods=methods)

    if output_prefix:
        Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)

        pickle_file = f"{output_prefix}_diversity_data.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump({
                'per_file_results': all_results,
                'aggregated_diversity': dict(aggregated_diversity)
            }, f)
        print(f"Data saved to {pickle_file}")

        summary_stats = calculate_summary_statistics(aggregated_diversity, methods=methods)
        json_file = f"{output_prefix}_diversity_summary.json"
        with open(json_file, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        print(f"Summary saved to {json_file}")

        txt_file = f"{output_prefix}_diversity_report.txt"
        generate_diversity_report(txt_file, summary_stats, processed, aggregated_diversity,
                                  methods=methods)
        print(f"Report saved to {txt_file}")

    return all_results, aggregated_diversity


def print_diversity_summary(aggregated_diversity, methods=DEFAULT_METHODS):
    """Print diversity summary statistics."""

    print("=" * 90)
    print("DIVERSITY SUMMARY - AVERAGE ACROSS ALL SYSTEMS")
    print("=" * 90)
    print()

    for method in methods:
        print(f"{method.upper()}")
        print("-" * 90)

        method_data = aggregated_diversity[method]

        print("  Distance diversity (Coefficient of Variation):")
        for key, values in method_data.items():
            if key.startswith('distance_cv_') and values:
                feature = key.replace('distance_cv_', '')
                mean, ci_low, ci_high = bootstrap_diversity_ci(values)
                print(f"    {feature:25s}: {mean:.4f} +/- {(ci_high-ci_low)/2:.4f} "
                      f"[{ci_low:.4f}, {ci_high:.4f}]")

        print("\n  Angular diversity (Circular Variance):")
        for key, values in method_data.items():
            if key.startswith('angular_variance_') and values:
                feature = key.replace('angular_variance_', '')
                mean, ci_low, ci_high = bootstrap_diversity_ci(values)
                print(f"    {feature:25s}: {mean:.4f} +/- {(ci_high-ci_low)/2:.4f} "
                      f"[{ci_low:.4f}, {ci_high:.4f}]")

        print()


def bootstrap_diversity_ci(values, n_bootstrap=1000):
    """Calculate bootstrap confidence interval for diversity metric."""
    if len(values) == 0:
        return 0, 0, 0

    bootstrap_means = []
    n = len(values)

    for _ in range(n_bootstrap):
        sample = random.choices(values, k=n)
        bootstrap_means.append(sum(sample) / len(sample))

    bootstrap_means.sort()
    alpha = 0.025
    lower_idx = int(alpha * n_bootstrap)
    upper_idx = int((1 - alpha) * n_bootstrap)

    mean = sum(values) / len(values)
    ci_lower = bootstrap_means[lower_idx]
    ci_upper = bootstrap_means[upper_idx]

    return mean, ci_lower, ci_upper


def calculate_summary_statistics(aggregated_diversity, methods=DEFAULT_METHODS, n_bootstrap=1000):
    """Calculate summary statistics with bootstrap confidence intervals."""

    summary = {}

    for method in methods:
        method_data = aggregated_diversity[method]
        summary[method] = {}

        for key, values in method_data.items():
            if values:
                mean, ci_lower, ci_upper = bootstrap_diversity_ci(values, n_bootstrap)

                summary[method][key] = {
                    'mean': mean,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'ci_width': ci_upper - ci_lower,
                    'min': min(values),
                    'max': max(values),
                    'n_systems': len(values)
                }

    return summary


def bootstrap_average_diversity_ci(aggregated_diversity, metric_prefix, n_bootstrap=1000,
                                 methods=DEFAULT_METHODS):
    """
    Calculate bootstrap CI for average diversity across multiple metrics.
    Uses system-level resampling to properly account for correlations between metrics.
    """
    method_results = {}

    for method in methods:
        metric_keys = [k for k in aggregated_diversity[method].keys()
                       if k.startswith(metric_prefix + '_')]

        if not metric_keys:
            continue

        n_systems = len(aggregated_diversity[method][metric_keys[0]])

        system_data = []
        for i in range(n_systems):
            system_values = []
            for key in metric_keys:
                if i < len(aggregated_diversity[method][key]):
                    system_values.append(aggregated_diversity[method][key][i])
            if system_values:
                system_data.append(system_values)

        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = random.choices(system_data, k=len(system_data))
            system_averages = [sum(sys_vals) / len(sys_vals) for sys_vals in sample]
            bootstrap_means.append(sum(system_averages) / len(system_averages))

        bootstrap_means.sort()
        alpha = 0.025
        lower_idx = int(alpha * n_bootstrap)
        upper_idx = int((1 - alpha) * n_bootstrap)

        actual_mean = sum(sum(sys_vals) / len(sys_vals)
                         for sys_vals in system_data) / len(system_data)

        method_results[method] = {
            'mean': actual_mean,
            'ci_lower': bootstrap_means[lower_idx],
            'ci_upper': bootstrap_means[upper_idx]
        }

    return method_results


def generate_diversity_report(output_file, summary_stats, processed, aggregated_diversity,
                              methods=DEFAULT_METHODS):
    """Generate diversity analysis report."""

    methods = tuple(methods)
    with open(output_file, 'w') as f:
        f.write("=" * 90 + "\n")
        f.write("DIVERSITY ANALYSIS OF ALL-CORRECT PREDICTIONS\n")
        f.write("=" * 90 + "\n\n")

        f.write(f"Total systems analyzed: {processed}\n")
        f.write(f"Analysis: Only predictions with all 8 labels correct\n")
        f.write(
            f"Selection: Systems where all {len(methods)} methods have >=10 all-correct predictions\n\n"
        )

        f.write("=" * 90 + "\n")
        f.write("DIVERSITY METRICS SUMMARY\n")
        f.write("=" * 90 + "\n\n")

        metric_types = [
            ('distance_cv', 'Distance Diversity (Coefficient of Variation)', 'Higher = more diverse'),
            ('angular_variance', 'Angular Diversity (Circular Variance)', 'Higher = more diverse')
        ]

        for metric_prefix, metric_name, interpretation in metric_types:
            f.write(f"\n{metric_name} ({interpretation}):\n")
            f.write("-" * 90 + "\n\n")

            features = set()
            for method in methods:
                for key in summary_stats.get(method, {}).keys():
                    if key.startswith(metric_prefix + '_'):
                        feature = key.replace(metric_prefix + '_', '')
                        features.add(feature)

            for feature in sorted(features):
                f.write(f"  {feature}:\n")

                method_values = []
                for method in methods:
                    key = f"{metric_prefix}_{feature}"
                    if key in summary_stats[method]:
                        stats = summary_stats[method][key]
                        method_values.append((method, stats['mean'], stats['ci_lower'], stats['ci_upper']))
                        f.write(f"    {method.upper():10s}: {stats['mean']:.4f} +/- {stats['ci_width']/2:.4f} "
                               f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}] "
                               f"(n={stats['n_systems']:4d})\n")

                if method_values:
                    most_diverse = max(method_values, key=lambda x: x[1])
                    least_diverse = min(method_values, key=lambda x: x[1])

                    overlap = not (most_diverse[3] < least_diverse[2] or
                                   least_diverse[3] < most_diverse[2])

                    if overlap:
                        f.write(f"    Most diverse: {most_diverse[0].upper()} "
                               f"(CIs overlap with {least_diverse[0].upper()} - "
                               f"difference may not be significant)\n")
                    else:
                        f.write(f"    Most diverse: {most_diverse[0].upper()} "
                               f"(significantly different from {least_diverse[0].upper()})\n")

                f.write("\n")

        f.write("\n")
        f.write("=" * 90 + "\n")
        f.write("COMPARATIVE RANKINGS\n")
        f.write("=" * 90 + "\n\n")

        f.write("Average Distance Diversity (Coefficient of Variation):\n")
        f.write("(Averaged across 4 distance metrics with bootstrap 95% CI)\n\n")

        distance_avg_ci = bootstrap_average_diversity_ci(
            aggregated_diversity, 'distance_cv', methods=methods
        )

        dist_rankings = sorted(distance_avg_ci.items(), key=lambda x: x[1]['mean'], reverse=True)
        for i, (method, stats) in enumerate(dist_rankings, 1):
            ci_width = stats['ci_upper'] - stats['ci_lower']
            f.write(f"  {i}. {method.upper():10s}: {stats['mean']:.4f} +/- {ci_width/2:.4f} "
                   f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]\n")

        f.write("\n\nAverage Angular Diversity (Circular Variance):\n")
        f.write("(Averaged across 8 angle metrics with bootstrap 95% CI)\n\n")

        angular_avg_ci = bootstrap_average_diversity_ci(
            aggregated_diversity, 'angular_variance', methods=methods
        )

        ang_rankings = sorted(angular_avg_ci.items(), key=lambda x: x[1]['mean'], reverse=True)
        for i, (method, stats) in enumerate(ang_rankings, 1):
            ci_width = stats['ci_upper'] - stats['ci_lower']
            f.write(f"  {i}. {method.upper():10s}: {stats['mean']:.4f} +/- {ci_width/2:.4f} "
                   f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]\n")

        f.write("\n\nKey findings:\n")
        if dist_rankings and ang_rankings:
            f.write(f"- Distance diversity ranking: "
                    f"{' > '.join([m.upper() for m, stats in dist_rankings])}\n")
            f.write(f"- Angular diversity ranking: "
                    f"{' > '.join([m.upper() for m, stats in ang_rankings])}\n")

            if len(dist_rankings) >= 2:
                top_dist = dist_rankings[0]
                second_dist = dist_rankings[1]
                if top_dist[1]['ci_lower'] > second_dist[1]['ci_upper']:
                    f.write(f"- {top_dist[0].upper()} is SIGNIFICANTLY more diverse than "
                           f"{second_dist[0].upper()} for distance metrics "
                           f"(non-overlapping 95% CIs)\n")

            if len(ang_rankings) >= 2:
                top_ang = ang_rankings[0]
                second_ang = ang_rankings[1]
                if top_ang[1]['ci_lower'] > second_ang[1]['ci_upper']:
                    f.write(f"- {top_ang[0].upper()} is SIGNIFICANTLY more diverse than "
                           f"{second_ang[0].upper()} for angular metrics "
                           f"(non-overlapping 95% CIs)\n")

            dist_leader = dist_rankings[0][0]
            ang_leader = ang_rankings[0][0]
            if dist_leader == ang_leader:
                f.write(f"- {dist_leader.upper()} is most diverse for both distance "
                       f"AND angular metrics\n")
            else:
                f.write(f"- {dist_leader.upper()} most diverse for distances, "
                       f"{ang_leader.upper()} most diverse for angles\n")

        f.write(f"\n- Lower diversity -> more consistent/stable predictions\n")
        f.write(f"- Higher diversity -> better conformational sampling\n")
        f.write(f"\nNote: Distance CV and angular variance are on different scales "
                f"and not combined.\n")
        f.write(f"Categorical labels excluded as all predictions match ground truth.\n")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze diversity of all-correct predictions'
    )
    parser.add_argument(
        '-i', '--input',
        help='Input directory with CSVs (from extract_correct_predictions.py diversity output)',
        default='csvs_for_diversity_analysis'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file prefix',
        default='diversity_results'
    )
    parser.add_argument(
        '--methods',
        type=str,
        default=','.join(DEFAULT_METHODS),
        help='Comma-separated methods (default: all four)',
    )

    args = parser.parse_args()

    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    mlist = tuple(x.strip().lower() for x in args.methods.split(',') if x.strip())
    if not mlist:
        raise SystemExit("--methods must list at least one method")

    aggregate_diversity_analysis(args.input, args.output, methods=mlist)


if __name__ == '__main__':
    main()

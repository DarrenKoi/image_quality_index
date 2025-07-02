from pathlib import Path
import json
from datetime import datetime
from CDSEM_IMAGE_QUALITY_ANALYZER import CDSEMImageAnalyzer
import numpy as np
import csv
import os


def analyze_sem_images(image_directory, output_dir="analysis_results"):
    """
    Complete analysis workflow for SEM images
    """
    # Convert to Path objects
    image_dir_path = Path(image_directory)
    output_path = Path(output_dir)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize analyzer
    analyzer = CDSEMImageAnalyzer(
        use_visualization=True,
        debug_dir=str(output_path / "debug_images")
    )

    # Get all TIFF images
    tiff_files = []
    for pattern in ['*.tif', '*.tiff', '*.TIF', '*.TIFF']:
        tiff_files.extend(image_dir_path.rglob(pattern))

    # Convert to strings for the analyzer
    tiff_file_paths = [str(f) for f in tiff_files]

    print(f"Found {len(tiff_files)} TIFF images in {image_dir_path}")

    if not tiff_files:
        print("No TIFF files found!")
        return [], {}

    # Step 1: Find best reference image
    print("\nStep 1: Finding best reference image...")
    reference_path = find_best_reference(tiff_files[:min(10, len(tiff_files))], analyzer)

    # Step 2: Full analysis with drift compensation
    print(f"\nStep 2: Analyzing all {len(tiff_files)} images...")
    all_results = analyzer.batch_analyze(
        tiff_file_paths,
        reference_path=str(reference_path),
        max_workers=4
    )

    # Step 3: Generate summary report
    summary = generate_summary(all_results, tiff_files, reference_path)

    # Save results
    save_results(all_results, summary, output_path)

    # Generate reports
    generate_csv_report(all_results, output_path / 'analysis_report.csv')
    generate_grouped_reports(tiff_files, all_results, output_path)

    print(f"\n✅ Analysis complete! Results saved to: {output_path.absolute()}")
    return all_results, summary


def find_best_reference(candidate_files, analyzer):
    """
    Find the best reference image based on quality metrics
    """
    best_quality = 0
    best_path = None

    for i, path in enumerate(candidate_files):
        print(f"  Checking {i + 1}/{len(candidate_files)}: {path.name}")
        result = analyzer.analyze_image(str(path))
        quality = result['quality_metrics']['overall_quality']

        if quality > best_quality:
            best_quality = quality
            best_path = path

    print(f"\nSelected reference: {best_path.name}")
    print(f"Reference quality: {best_quality:.1f}/100")

    return best_path


def generate_summary(results, tiff_files, reference_path):
    """
    Generate summary statistics
    """
    summary = {
        'analysis_date': datetime.now().isoformat(),
        'total_images': len(tiff_files),
        'reference_image': str(reference_path),
        'reference_image_name': reference_path.name,
        'statistics': calculate_statistics(results),
        'pattern_distribution': count_patterns(results),
        'quality_distribution': quality_histogram(results)
    }

    return summary


def save_results(results, summary, output_path):
    """
    Save analysis results to JSON files
    """
    # Full results
    with open(output_path / 'full_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Summary
    with open(output_path / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


def calculate_statistics(results):
    """
    Calculate statistics from results
    """
    valid_results = [r for r in results if 'quality_metrics' in r and 'error' not in r]

    if not valid_results:
        return {}

    qualities = [r['quality_metrics']['overall_quality'] for r in valid_results]
    contrasts = [r['quality_metrics']['contrast'] for r in valid_results]
    sharpnesses = [r['quality_metrics']['sharpness'] for r in valid_results]
    snrs = [r['quality_metrics']['snr_db'] for r in valid_results]

    return {
        'quality': {
            'mean': float(np.mean(qualities)),
            'std': float(np.std(qualities)),
            'min': float(np.min(qualities)),
            'max': float(np.max(qualities))
        },
        'contrast': {
            'mean': float(np.mean(contrasts)),
            'std': float(np.std(contrasts)),
            'min': float(np.min(contrasts)),
            'max': float(np.max(contrasts))
        },
        'sharpness': {
            'mean': float(np.mean(sharpnesses)),
            'std': float(np.std(sharpnesses)),
            'min': float(np.min(sharpnesses)),
            'max': float(np.max(sharpnesses))
        },
        'snr': {
            'mean': float(np.mean(snrs)),
            'std': float(np.std(snrs)),
            'min': float(np.min(snrs)),
            'max': float(np.max(snrs))
        }
    }


def count_patterns(results):
    """
    Count pattern types detected
    """
    pattern_counts = {}
    for r in results:
        if 'pattern_analysis' in r and 'error' not in r:
            pattern = r['pattern_analysis']['pattern_type']
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    return pattern_counts


def quality_histogram(results):
    """
    Create quality score distribution
    """
    valid_results = [r for r in results if 'quality_metrics' in r and 'error' not in r]
    qualities = [r['quality_metrics']['overall_quality'] for r in valid_results]

    bins = [0, 20, 40, 60, 80, 100]
    hist = {f"{bins[i]}-{bins[i + 1]}": 0 for i in range(len(bins) - 1)}

    for q in qualities:
        for i in range(len(bins) - 1):
            if bins[i] <= q < bins[i + 1]:
                hist[f"{bins[i]}-{bins[i + 1]}"] += 1
                break

    return hist


def generate_csv_report(results, output_path):
    """
    Generate CSV report
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            'Image Path', 'Image Name', 'Directory', 'Overall Quality',
            'Contrast', 'Sharpness', 'SNR (dB)', 'Focus Quality',
            'Noise Std', 'Pattern Type', 'Average CD (nm)',
            'Drift X', 'Drift Y', 'Drift Magnitude'
        ])

        # Data rows
        for r in results:
            if 'quality_metrics' in r and 'error' not in r:
                image_path = Path(r['image_path'])
                drift = r.get('drift_analysis', {})
                pattern = r.get('pattern_analysis', {})
                line_space = pattern.get('line_space_analysis', {})

                writer.writerow([
                    str(image_path),
                    image_path.name,
                    image_path.parent.name,
                    f"{r['quality_metrics']['overall_quality']:.2f}",
                    f"{r['quality_metrics']['contrast']:.2f}",
                    f"{r['quality_metrics']['sharpness']:.2f}",
                    f"{r['quality_metrics']['snr_db']:.2f}",
                    f"{r['quality_metrics']['focus_quality']:.4f}",
                    f"{r['quality_metrics']['noise_std']:.2f}",
                    pattern.get('pattern_type', 'N/A'),
                    f"{line_space.get('average_cd_nm', 0):.2f}",
                    f"{drift.get('shift_x', 0):.2f}",
                    f"{drift.get('shift_y', 0):.2f}",
                    f"{drift.get('drift_magnitude', 0):.2f}"
                ])


def generate_grouped_reports(tiff_files, results, output_path):
    """
    Generate reports grouped by directory
    """
    grouped_path = output_path / "grouped_reports"
    grouped_path.mkdir(exist_ok=True)

    # Create mapping of path to result
    path_to_result = {}
    for result in results:
        if 'error' not in result:
            path_to_result[result['image_path']] = result

    # Group by parent directory
    directory_groups = {}
    for tiff_path in tiff_files:
        parent = tiff_path.parent
        if parent not in directory_groups:
            directory_groups[parent] = []
        directory_groups[parent].append(tiff_path)

    # Generate report for each directory
    for directory, files in directory_groups.items():
        dir_name = directory.name or "root"
        report_data = []

        for file_path in files:
            if str(file_path) in path_to_result:
                result = path_to_result[str(file_path)]
                report_data.append({
                    'filename': file_path.name,
                    'quality': result['quality_metrics']['overall_quality'],
                    'pattern': result['pattern_analysis']['pattern_type'],
                    'contrast': result['quality_metrics']['contrast'],
                    'sharpness': result['quality_metrics']['sharpness'],
                    'snr': result['quality_metrics']['snr_db']
                })

        if report_data:
            # Calculate directory statistics
            dir_summary = {
                'directory': str(directory),
                'file_count': len(report_data),
                'statistics': {
                    'quality': {
                        'mean': float(np.mean([d['quality'] for d in report_data])),
                        'std': float(np.std([d['quality'] for d in report_data])),
                        'min': float(np.min([d['quality'] for d in report_data])),
                        'max': float(np.max([d['quality'] for d in report_data]))
                    }
                },
                'files': report_data
            }

            # Save directory report
            safe_dir_name = dir_name.replace('/', '_').replace('\\', '_')
            with open(grouped_path / f"{safe_dir_name}_report.json", 'w') as f:
                json.dump(dir_summary, f, indent=2)


# Quick analysis function for testing
def quick_analyze(image_path, reference_path=None):
    """
    Quick analysis of a single image or small set
    """
    analyzer = CDSEMImageAnalyzer(use_visualization=True)

    if isinstance(image_path, (str, Path)):
        # Single image
        result = analyzer.analyze_image(str(image_path),
                                        reference_image_path=str(reference_path) if reference_path else None)
        print_single_result(result)
        return result
    else:
        # Multiple images
        results = []
        for path in image_path:
            result = analyzer.analyze_image(str(path),
                                            reference_image_path=str(reference_path) if reference_path else None)
            results.append(result)
            print_single_result(result)
        return results


def print_single_result(result):
    """
    Print formatted result for a single image
    """
    if 'error' in result:
        print(f"❌ Error analyzing {result['image_path']}: {result['error']}")
    else:
        print(f"\n📊 Analysis Results for: {Path(result['image_path']).name}")
        print(f"   Overall Quality: {result['quality_metrics']['overall_quality']:.1f}/100")
        print(f"   Contrast: {result['quality_metrics']['contrast']:.1f}")
        print(f"   Sharpness: {result['quality_metrics']['sharpness']:.1f}")
        print(f"   SNR: {result['quality_metrics']['snr_db']:.1f} dB")
        print(f"   Pattern Type: {result['pattern_analysis']['pattern_type']}")

        if result.get('drift_analysis'):
            drift = result['drift_analysis']
            print(
                f"   Drift: X={drift['shift_x']:.1f}, Y={drift['shift_y']:.1f}, Magnitude={drift['drift_magnitude']:.1f}")


# Main execution
if __name__ == "__main__":
    # Example 1: Analyze a directory
    image_directory = Path("test_images")  # Change to your directory

    if image_directory.exists():
        results, summary = analyze_sem_images(image_directory)

        print("\n📊 Final Summary:")
        print(f"Total images: {summary['total_images']}")
        if summary['statistics']:
            print(
                f"Average quality: {summary['statistics']['quality']['mean']:.1f} ± {summary['statistics']['quality']['std']:.1f}")
            print(
                f"Quality range: {summary['statistics']['quality']['min']:.1f} - {summary['statistics']['quality']['max']:.1f}")
    else:
        print(f"Directory not found: {image_directory}")
        print("Creating test images...")

        # Create dummy images for testing
        analyzer = CDSEMImageAnalyzer()
        image_directory.mkdir(exist_ok=True)

        test_images = [
            image_directory / "test_line_pattern.tif",
            image_directory / "test_contact_holes.tif",
            image_directory / "test_mixed_pattern.tif"
        ]

        for img_path in test_images:
            analyzer._create_dummy_image(str(img_path))

        # Now analyze
        results, summary = analyze_sem_images(image_directory)

    # Example 2: Quick analysis of specific images
    print("\n" + "=" * 50)
    print("Quick analysis example:")

    # Single image
    single_image = Path("test_images/test_line_pattern.tif")
    if single_image.exists():
        quick_result = quick_analyze(single_image)

    # Multiple images with reference
    all_tifs = list(Path("test_images").glob("*.tif"))
    if len(all_tifs) > 1:
        print("\nAnalyzing with reference image...")
        quick_analyze(all_tifs[1], reference_path=all_tifs[0])
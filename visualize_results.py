#!/usr/bin/env python3
"""
GPU Merkle Tree Benchmark Visualization
Analyzes and visualizes performance metrics from GPU benchmark tests
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import sys

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

def load_results(csv_file="gpu_benchmark_results.csv"):
    """Load benchmark results from CSV file"""
    if not Path(csv_file).exists():
        print(f"Error: {csv_file} not found. Run the benchmark first!")
        sys.exit(1)

    df = pd.read_csv(csv_file)
    return df

def analyze_performance_ceiling(df, output_dir="plots"):
    """Analyze where performance plateaus and identify optimal chunk sizes"""
    gpu_df = df[df['Implementation'] == 'GPU'].copy()

    if gpu_df.empty:
        return

    print("\n" + "="*70)
    print("PERFORMANCE CEILING ANALYSIS")
    print("="*70)

    tests_with_chunks = gpu_df.groupby('Test')['ChunkSize'].nunique()
    tests_with_chunks = tests_with_chunks[tests_with_chunks > 1].index.tolist()

    for test in tests_with_chunks:
        test_df = gpu_df[gpu_df['Test'] == test].sort_values('ChunkSize')

        if len(test_df) < 3:
            continue

        # Find where improvement drops below 1%
        plateau_point = None
        for i in range(1, len(test_df)):
            prev_time = test_df.iloc[i-1]['TimeMS']
            curr_time = test_df.iloc[i]['TimeMS']
            improvement = (prev_time - curr_time) / prev_time * 100

            if improvement < 1.0 and test_df.iloc[i]['ChunkSize'] > 16384:
                plateau_point = test_df.iloc[i]['ChunkSize']
                break

        min_time_idx = test_df['TimeMS'].idxmin()
        optimal_chunk = test_df.loc[min_time_idx, 'ChunkSize']
        optimal_time = test_df.loc[min_time_idx, 'TimeMS']
        optimal_speedup = test_df.loc[min_time_idx, 'Speedup']

        print(f"\n{test}:")
        print(f"  Optimal chunk: {optimal_chunk:,} items")
        print(f"  Best time: {optimal_time:.2f} ms ({optimal_speedup:.2f}x speedup)")

        if plateau_point:
            print(f"  ⚠️  Performance plateau at: {plateau_point:,} items")
            print(f"      (further increases yield < 1% improvement)")

    print("\n" + "="*70 + "\n")

def plot_comprehensive_chunk_comparison(df, output_dir="plots"):
    """Create comprehensive comparison chart with all chunk sizes in different colors"""
    Path(output_dir).mkdir(exist_ok=True)

    # Filter for GPU results only
    gpu_df = df[df['Implementation'] == 'GPU'].copy()

    if gpu_df.empty:
        print("No GPU results found for chunk size optimization")
        return

    # Get all tests that have chunk size variation
    tests_with_chunks = gpu_df.groupby('Test')['ChunkSize'].nunique()
    tests_with_chunks = tests_with_chunks[tests_with_chunks > 1].index.tolist()

    if not tests_with_chunks:
        print("No chunk size variation found")
        return

    # Define color palette for chunk sizes (extended range support)
    chunk_sizes = sorted(gpu_df['ChunkSize'].unique())
    colors = plt.cm.rainbow(np.linspace(0, 1, len(chunk_sizes)))
    chunk_color_map = dict(zip(chunk_sizes, colors))

    # Create comprehensive figure (scale with number of tests)
    num_tests = len(tests_with_chunks)
    fig_height = max(12, 8 + num_tests * 0.5)  # Scale figure with test count
    fig = plt.figure(figsize=(22, fig_height))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    fig.suptitle(f'Comprehensive GPU Chunk Size Comparison ({num_tests} Test Scenarios)',
                 fontsize=18, fontweight='bold', y=0.98)

    # Plot 1: Execution Time Comparison (all tests, all chunks)
    ax1 = fig.add_subplot(gs[0, :])
    for chunk_size in chunk_sizes:
        chunk_df = gpu_df[gpu_df['ChunkSize'] == chunk_size]
        if not chunk_df.empty:
            ax1.plot(range(len(tests_with_chunks)),
                    [chunk_df[chunk_df['Test'] == t]['TimeMS'].values[0]
                     if t in chunk_df['Test'].values else np.nan
                     for t in tests_with_chunks],
                    marker='o', linewidth=2.5, markersize=8,
                    color=chunk_color_map[chunk_size],
                    label=f'{int(chunk_size):,} items')

    ax1.set_xlabel('Test Scenario', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Execution Time (ms)', fontweight='bold', fontsize=12)
    ax1.set_title('Execution Time: All Chunk Sizes Across Tests', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(tests_with_chunks)))
    ax1.set_xticklabels([t.replace(' ', '\n') for t in tests_with_chunks],
                        rotation=0, ha='center', fontsize=9)
    ax1.legend(title='Chunk Size', bbox_to_anchor=(1.02, 1), loc='upper left',
              ncol=1, fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Plot 2: Speedup Comparison (filter out zeros from tests without CPU baseline)
    ax2 = fig.add_subplot(gs[1, 0])
    for chunk_size in chunk_sizes:
        chunk_df = gpu_df[gpu_df['ChunkSize'] == chunk_size]
        if not chunk_df.empty:
            speedups = []
            for t in tests_with_chunks:
                if t in chunk_df['Test'].values:
                    speedup = chunk_df[chunk_df['Test'] == t]['Speedup'].values[0]
                    # Replace 0 with NaN (tests without CPU baseline)
                    speedups.append(speedup if speedup > 0 else np.nan)
                else:
                    speedups.append(np.nan)

            ax2.plot(range(len(tests_with_chunks)), speedups,
                    marker='s', linewidth=2, markersize=7,
                    color=chunk_color_map[chunk_size],
                    label=f'{int(chunk_size):,}')

    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2,
                alpha=0.5, label='CPU Baseline')
    ax2.set_xlabel('Test Scenario', fontweight='bold')
    ax2.set_ylabel('Speedup vs CPU', fontweight='bold')
    ax2.set_title('Speedup Comparison (excludes tests without CPU baseline)',
                  fontsize=13, fontweight='bold')
    ax2.set_xticks(range(len(tests_with_chunks)))
    ax2.set_xticklabels([t.split('(')[0].strip() for t in tests_with_chunks],
                        rotation=45, ha='right', fontsize=8)
    ax2.legend(title='Chunk Size', fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Throughput Comparison
    ax3 = fig.add_subplot(gs[1, 1])
    for chunk_size in chunk_sizes:
        chunk_df = gpu_df[gpu_df['ChunkSize'] == chunk_size]
        if not chunk_df.empty:
            ax3.plot(range(len(tests_with_chunks)),
                    [chunk_df[chunk_df['Test'] == t]['ThroughputMBps'].values[0]
                     if t in chunk_df['Test'].values else np.nan
                     for t in tests_with_chunks],
                    marker='^', linewidth=2, markersize=7,
                    color=chunk_color_map[chunk_size],
                    label=f'{int(chunk_size):,}')

    ax3.set_xlabel('Test Scenario', fontweight='bold')
    ax3.set_ylabel('Throughput (MB/s)', fontweight='bold')
    ax3.set_title('Data Throughput Comparison', fontsize=13, fontweight='bold')
    ax3.set_xticks(range(len(tests_with_chunks)))
    ax3.set_xticklabels([t.split('(')[0].strip() for t in tests_with_chunks],
                        rotation=45, ha='right', fontsize=8)
    ax3.legend(title='Chunk Size', fontsize=8, ncol=2)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Heatmap of Speedups
    ax4 = fig.add_subplot(gs[2, :])

    # Prepare data for heatmap (handle tests without CPU baseline)
    heatmap_data = []
    for test in tests_with_chunks:
        row = []
        for chunk_size in chunk_sizes:
            chunk_df = gpu_df[(gpu_df['Test'] == test) & (gpu_df['ChunkSize'] == chunk_size)]
            if not chunk_df.empty:
                speedup = chunk_df['Speedup'].values[0]
                # Replace 0 with NaN (tests without CPU baseline like 100M)
                row.append(speedup if speedup > 0 else np.nan)
            else:
                row.append(np.nan)
        heatmap_data.append(row)

    heatmap_data = np.array(heatmap_data)

    # Use masked array to handle NaN values properly
    masked_data = np.ma.masked_invalid(heatmap_data)

    im = ax4.imshow(masked_data, cmap='RdYlGn', aspect='auto', interpolation='nearest',
                    vmin=0, vmax=np.nanmax(heatmap_data) if np.any(~np.isnan(heatmap_data)) else 10)

    # Set ticks and labels
    ax4.set_xticks(np.arange(len(chunk_sizes)))
    ax4.set_yticks(np.arange(len(tests_with_chunks)))
    ax4.set_xticklabels([f'{int(cs):,}' for cs in chunk_sizes], rotation=45, ha='right', fontsize=9)
    ax4.set_yticklabels([t.split('(')[0].strip() for t in tests_with_chunks], fontsize=9)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Speedup vs CPU', rotation=270, labelpad=20, fontweight='bold')

    # Add text annotations
    for i in range(len(tests_with_chunks)):
        for j in range(len(chunk_sizes)):
            if not np.isnan(heatmap_data[i, j]):
                text = ax4.text(j, i, f'{heatmap_data[i, j]:.1f}x',
                               ha="center", va="center", color="black", fontsize=8)
            else:
                # Mark tests without baseline
                text = ax4.text(j, i, 'N/A',
                               ha="center", va="center", color="gray",
                               fontsize=7, fontstyle='italic')

    ax4.set_xlabel('Chunk Size (items)', fontweight='bold', fontsize=12)
    ax4.set_ylabel('Test Scenario', fontweight='bold', fontsize=12)
    ax4.set_title('Speedup Heatmap: Find Optimal Chunk Size per Workload\n' +
                  '(Gray "N/A" = no CPU baseline for comparison)',
                  fontsize=13, fontweight='bold')

    plt.tight_layout()
    output_file = f"{output_dir}/comprehensive_chunk_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_chunk_size_optimization(df, output_dir="plots"):
    """Plot GPU chunk size optimization results"""
    Path(output_dir).mkdir(exist_ok=True)

    # Filter for GPU results only
    gpu_df = df[df['Implementation'] == 'GPU'].copy()

    if gpu_df.empty:
        print("No GPU results found for chunk size optimization")
        return

    # Group by test name
    tests = gpu_df['Test'].unique()

    for test in tests:
        test_df = gpu_df[gpu_df['Test'] == test]

        if test_df['ChunkSize'].nunique() < 2:
            continue  # Skip if not chunk size variation

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'GPU Chunk Size Optimization: {test}', fontsize=16, fontweight='bold')

        # Plot 1: Total Time vs Chunk Size
        ax1 = axes[0, 0]
        ax1.plot(test_df['ChunkSize'], test_df['TimeMS'], marker='o', linewidth=2, markersize=8)
        ax1.set_xlabel('Chunk Size (items)', fontweight='bold')
        ax1.set_ylabel('Total Time (ms)', fontweight='bold')
        ax1.set_title('Execution Time vs Chunk Size')
        ax1.set_xscale('log', base=2)
        ax1.grid(True, alpha=0.3)

        # Mark optimal point
        optimal_idx = test_df['TimeMS'].idxmin()
        optimal_chunk = test_df.loc[optimal_idx, 'ChunkSize']
        optimal_time = test_df.loc[optimal_idx, 'TimeMS']
        ax1.axvline(optimal_chunk, color='red', linestyle='--', alpha=0.5, label=f'Optimal: {optimal_chunk}')
        ax1.legend()

        # Plot 2: Speedup vs Chunk Size
        ax2 = axes[0, 1]
        ax2.plot(test_df['ChunkSize'], test_df['Speedup'], marker='s', linewidth=2,
                markersize=8, color='green')
        ax2.set_xlabel('Chunk Size (items)', fontweight='bold')
        ax2.set_ylabel('Speedup vs CPU Baseline', fontweight='bold')
        ax2.set_title('Speedup vs Chunk Size')
        ax2.set_xscale('log', base=2)
        ax2.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='CPU Baseline')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Plot 3: Time Breakdown (Stacked)
        ax3 = axes[1, 0]
        x_pos = np.arange(len(test_df))

        ax3.bar(x_pos, test_df['MemoryH2DMS'], label='Host→Device', alpha=0.8)
        ax3.bar(x_pos, test_df['KernelMS'], bottom=test_df['MemoryH2DMS'],
               label='Kernel Execution', alpha=0.8)
        ax3.bar(x_pos, test_df['MemoryD2HMS'],
               bottom=test_df['MemoryH2DMS'] + test_df['KernelMS'],
               label='Device→Host', alpha=0.8)

        ax3.set_xlabel('Chunk Size (items)', fontweight='bold')
        ax3.set_ylabel('Time (ms)', fontweight='bold')
        ax3.set_title('Time Breakdown by Operation')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([f'{int(c)}' for c in test_df['ChunkSize']], rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')

        # Plot 4: Throughput vs Chunk Size
        ax4 = axes[1, 1]
        ax4.plot(test_df['ChunkSize'], test_df['ThroughputMBps'], marker='^',
                linewidth=2, markersize=8, color='purple')
        ax4.set_xlabel('Chunk Size (items)', fontweight='bold')
        ax4.set_ylabel('Throughput (MB/s)', fontweight='bold')
        ax4.set_title('Data Throughput vs Chunk Size')
        ax4.set_xscale('log', base=2)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = f"{output_dir}/chunk_optimization_{test.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

def plot_workload_comparison(df, output_dir="plots"):
    """Plot workload comparison across implementations"""
    Path(output_dir).mkdir(exist_ok=True)

    # Get tests with multiple implementations
    implementation_counts = df.groupby('Test')['Implementation'].nunique()
    comparison_tests = implementation_counts[implementation_counts > 1].index

    if len(comparison_tests) == 0:
        print("No comparison tests found")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Workload Comparison: CPU vs GPU vs Hybrid', fontsize=16, fontweight='bold')

    # Prepare data for comparison
    comparison_df = df[df['Test'].isin(comparison_tests)].copy()

    # Group by test and implementation, take first entry (in case of duplicates)
    comparison_df = comparison_df.groupby(['Test', 'Implementation']).first().reset_index()

    # Plot 1: Execution Time Comparison
    ax1 = axes[0, 0]
    pivot_time = comparison_df.pivot(index='Test', columns='Implementation', values='TimeMS')
    pivot_time.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_ylabel('Execution Time (ms)', fontweight='bold')
    ax1.set_xlabel('Test', fontweight='bold')
    ax1.set_title('Execution Time Comparison')
    ax1.legend(title='Implementation')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Speedup Comparison
    ax2 = axes[0, 1]
    pivot_speedup = comparison_df.pivot(index='Test', columns='Implementation', values='Speedup')
    pivot_speedup.plot(kind='bar', ax=ax2, width=0.8)
    ax2.set_ylabel('Speedup vs Sequential', fontweight='bold')
    ax2.set_xlabel('Test', fontweight='bold')
    ax2.set_title('Speedup Comparison')
    ax2.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='Baseline')
    ax2.legend(title='Implementation')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Throughput Comparison
    ax3 = axes[1, 0]
    pivot_throughput = comparison_df.pivot(index='Test', columns='Implementation', values='ThroughputMBps')
    pivot_throughput.plot(kind='bar', ax=ax3, width=0.8)
    ax3.set_ylabel('Throughput (MB/s)', fontweight='bold')
    ax3.set_xlabel('Test', fontweight='bold')
    ax3.set_title('Data Throughput Comparison')
    ax3.legend(title='Implementation')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Efficiency (Speedup per MB)
    ax4 = axes[1, 1]
    comparison_df['EfficiencyScore'] = comparison_df['Speedup'] / (comparison_df['TotalBytes'] / 1024 / 1024)
    pivot_efficiency = comparison_df.pivot(index='Test', columns='Implementation', values='EfficiencyScore')
    pivot_efficiency.plot(kind='bar', ax=ax4, width=0.8)
    ax4.set_ylabel('Efficiency (Speedup per MB)', fontweight='bold')
    ax4.set_xlabel('Test', fontweight='bold')
    ax4.set_title('Implementation Efficiency')
    ax4.legend(title='Implementation')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_file = f"{output_dir}/workload_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_memory_overhead_analysis(df, output_dir="plots"):
    """Analyze memory transfer overhead for GPU implementations"""
    Path(output_dir).mkdir(exist_ok=True)

    gpu_df = df[df['Implementation'] == 'GPU'].copy()

    if gpu_df.empty or gpu_df['MemoryH2DMS'].sum() == 0:
        print("No GPU memory transfer data available")
        return

    # Calculate percentages
    gpu_df['TransferPercent'] = ((gpu_df['MemoryH2DMS'] + gpu_df['MemoryD2HMS']) / gpu_df['TimeMS']) * 100
    gpu_df['ComputePercent'] = (gpu_df['KernelMS'] / gpu_df['TimeMS']) * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('GPU Memory Transfer Overhead Analysis', fontsize=16, fontweight='bold')

    # Plot 1: Time breakdown pie chart (average across all tests)
    ax1 = axes[0]
    avg_h2d = gpu_df['MemoryH2DMS'].mean()
    avg_kernel = gpu_df['KernelMS'].mean()
    avg_d2h = gpu_df['MemoryD2HMS'].mean()

    labels = ['Host→Device', 'Kernel Execution', 'Device→Host']
    sizes = [avg_h2d, avg_kernel, avg_d2h]
    colors = ['#ff9999', '#66b3ff', '#99ff99']

    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Average Time Distribution')

    # Plot 2: Transfer overhead vs data size
    ax2 = axes[1]
    data_size_mb = gpu_df['TotalBytes'] / (1024 * 1024)
    ax2.scatter(data_size_mb, gpu_df['TransferPercent'], s=100, alpha=0.6)
    ax2.set_xlabel('Data Size (MB)', fontweight='bold')
    ax2.set_ylabel('Transfer Overhead (%)', fontweight='bold')
    ax2.set_title('Transfer Overhead vs Data Size')
    ax2.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(data_size_mb, gpu_df['TransferPercent'], 1)
    p = np.poly1d(z)
    ax2.plot(data_size_mb, p(data_size_mb), "r--", alpha=0.5, label='Trend')
    ax2.legend()

    plt.tight_layout()
    output_file = f"{output_dir}/memory_overhead_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def generate_summary_report(df, output_file="benchmark_summary.txt"):
    """Generate a text summary report"""
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("GPU MERKLE TREE BENCHMARK SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")

        # Overall statistics
        f.write("Overall Statistics:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total tests run: {len(df)}\n")
        f.write(f"Implementations tested: {', '.join(df['Implementation'].unique())}\n")
        f.write(f"Test scenarios: {df['Test'].nunique()}\n")
        f.write(f"Total data processed: {df['TotalBytes'].sum() / (1024**3):.2f} GB\n\n")

        # Best speedups
        f.write("Best Speedups:\n")
        f.write("-" * 80 + "\n")
        best_speedups = df.nlargest(5, 'Speedup')[['Test', 'Implementation', 'Speedup', 'TimeMS']]
        f.write(best_speedups.to_string(index=False))
        f.write("\n\n")

        # Optimal chunk sizes (for GPU tests)
        gpu_df = df[df['Implementation'] == 'GPU']
        if not gpu_df.empty and gpu_df['ChunkSize'].nunique() > 1:
            f.write("Optimal Chunk Sizes by Test:\n")
            f.write("-" * 80 + "\n")
            for test in gpu_df['Test'].unique():
                test_df = gpu_df[gpu_df['Test'] == test]
                if test_df['ChunkSize'].nunique() > 1:
                    optimal_idx = test_df['TimeMS'].idxmin()
                    optimal_chunk = test_df.loc[optimal_idx, 'ChunkSize']
                    optimal_speedup = test_df.loc[optimal_idx, 'Speedup']
                    f.write(f"{test}: {int(optimal_chunk)} items (Speedup: {optimal_speedup:.2f}x)\n")
            f.write("\n")

        # Implementation comparison
        f.write("Average Performance by Implementation:\n")
        f.write("-" * 80 + "\n")
        impl_avg = df.groupby('Implementation').agg({
            'TimeMS': 'mean',
            'Speedup': 'mean',
            'ThroughputMBps': 'mean'
        }).round(2)
        f.write(impl_avg.to_string())
        f.write("\n\n")

        f.write("=" * 80 + "\n")
        f.write("End of Report\n")
        f.write("=" * 80 + "\n")

    print(f"\nSummary report saved to: {output_file}")

def main():
    print("GPU Merkle Tree Benchmark Visualization")
    print("=" * 60)

    # Load results
    print("\nLoading benchmark results...")
    df = load_results()
    print(f"Loaded {len(df)} test results")

    # Analyze performance ceiling
    print("\nAnalyzing performance ceiling...")
    analyze_performance_ceiling(df)

    # Generate plots
    print("\nGenerating visualizations...")
    print("  Creating comprehensive chunk comparison chart...")
    plot_comprehensive_chunk_comparison(df)
    print("  Creating individual chunk optimization plots...")
    plot_chunk_size_optimization(df)
    print("  Creating workload comparison plots...")
    plot_workload_comparison(df)
    print("  Creating memory overhead analysis...")
    plot_memory_overhead_analysis(df)

    # Generate summary report
    print("\nGenerating summary report...")
    generate_summary_report(df)

    print("\n" + "=" * 60)
    print("Visualization complete! Check the 'plots' directory for images.")
    print("\nKey Output Files:")
    print("  📊 comprehensive_chunk_comparison.png - MAIN CHART (all chunk sizes)")
    print("  📈 chunk_optimization_*.png - Individual test details")
    print("  📊 workload_comparison.png - Implementation comparison")
    print("  🔍 memory_overhead_analysis.png - Transfer vs compute")
    print("  📄 benchmark_summary.txt - Text report with optimal configs")
    print("=" * 60)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
GPU Merkle Tree Benchmark Visualization
Creates 3 publication-quality plots:
1. CPU Comparison (Sequential vs Multithread 16T)
2. GPU Chunk Size Optimization Study
3. 4-Way Implementation Comparison (Sequential, Multithread, GPU Standard, GPU Adaptive)
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Set professional style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Data from benchmark (excluding structured dataset and hybrid)
workload_data = {
    'Small (1K x 100B)': {
        'Sequential': 10.21,
        'Multithread': 11.98,
        'GPU': 71.67,
        'size_mb': 0.095
    },
    'Medium (10K x 50-500B)': {
        'Sequential': 271.57,
        'Multithread': 283.07,
        'GPU': 22.53,
        'size_mb': 2.62
    },
    'Large (100K x 256B)': {
        'Sequential': 2587.99,
        'Multithread': 2653.08,
        'GPU': 211.04,
        'size_mb': 24.41
    },
    'Very Large (1M)': {
        'Sequential': None,  # Skipped
        'Multithread': None,  # Skipped
        'GPU': 2417.88,
        'Adaptive': 2276.56,  # From chunk optimization (best result)
        'size_mb': 143.05
    },
    'Extreme (10M)': {
        'Sequential': None,  # Skipped
        'Multithread': None,  # Skipped
        'GPU': 29591.08,
        'Adaptive': 22081.72,  # From chunk optimization (8K chunk)
        'size_mb': 1830.0
    }
}

# GPU Chunk optimization data (10K dataset)
chunk_opt_10k = {
    'chunk_sizes': [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576],
    'time_ms': [41.28, 30.27, 25.24, 22.85, 21.64, 21.10, 21.37, 21.03, 20.99, 20.99, 20.91, 21.56, 20.94, 21.01],
    'speedup': [6.72, 9.16, 10.99, 12.14, 12.82, 13.15, 12.98, 13.19, 13.21, 13.21, 13.27, 12.87, 13.25, 13.20],
}

# GPU Chunk optimization data (100K dataset)
chunk_opt_100k = {
    'chunk_sizes': [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576],
    'time_ms': [367.03, 291.65, 246.27, 224.16, 212.11, 206.87, 204.94, 204.30, 206.13, 207.58, 208.23, 207.75, 208.50, 208.02],
    'speedup': [7.10, 8.93, 10.57, 11.62, 12.28, 12.59, 12.71, 12.75, 12.63, 12.55, 12.51, 12.54, 12.49, 12.52],
}

# GPU Chunk optimization data (10M dataset)
chunk_opt_10m = {
    'chunk_sizes': [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576],
    'time_ms': [41205.70, 30793.95, 26481.06, 24482.73, 23220.53, 22473.94, 22081.72, 22082.15, 22216.39, 22369.91, 22395.59, 22311.92, 22363.82, 22374.78],
    'speedup': [5.00, 6.69, 7.78, 8.41, 8.87, 9.16, 9.32, 9.32, 9.27, 9.20, 9.19, 9.23, 9.21, 9.20],
}


def plot_cpu_comparison():
    """
    Plot 1: CPU Comparison - Sequential vs Multithread
    Shows performance of CPU-only implementations with 3 subplots:
    - Bar chart: Execution time comparison
    - Line plot: Throughput vs dataset size
    - Line plot: Throughput vs thread count (thread scaling analysis)
    """
    fig = plt.figure(figsize=(20, 6))
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2)
    ax3 = plt.subplot(1, 3, 3)

    # Filter workloads with CPU data
    workloads = ['Small (1K x 100B)', 'Medium (10K x 50-500B)', 'Large (100K x 256B)']
    workload_labels = [w.split('(')[0].strip() for w in workloads]

    seq_times = [workload_data[w]['Sequential'] for w in workloads]
    mt_times = [workload_data[w]['Multithread'] for w in workloads]
    data_sizes = [workload_data[w]['size_mb'] for w in workloads]

    x = np.arange(len(workload_labels))
    width = 0.35

    # Left plot: Bar comparison
    bars1 = ax1.bar(x - width/2, seq_times, width, label='Sequential', alpha=0.85, color='#FF6B6B')
    bars2 = ax1.bar(x + width/2, mt_times, width, label='Multithread (16T)', alpha=0.85, color='#4ECDC4')

    ax1.set_ylabel('Execution Time (ms)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Workload', fontsize=13, fontweight='bold')
    ax1.set_title('CPU Implementation Comparison\n(Sequential vs Multithread)',
                  fontsize=14, fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(workload_labels, fontsize=11)
    ax1.legend(fontsize=12, loc='upper left')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_yscale('log')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Middle plot: Throughput comparison
    seq_throughput = [data_sizes[i] / (seq_times[i] / 1000) for i in range(len(data_sizes))]
    mt_throughput = [data_sizes[i] / (mt_times[i] / 1000) for i in range(len(data_sizes))]

    ax2.plot(data_sizes, seq_throughput, 'o-', linewidth=3, markersize=12,
             label='Sequential', color='#FF6B6B')
    ax2.plot(data_sizes, mt_throughput, 's-', linewidth=3, markersize=12,
             label='Multithread (16T)', color='#4ECDC4')

    ax2.set_xlabel('Dataset Size (MB)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Throughput (MB/s)', fontsize=13, fontweight='bold')
    ax2.set_title('CPU Throughput vs Dataset Size',
                  fontsize=14, fontweight='bold', pad=20)
    ax2.set_xscale('log')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Annotate interesting points
    max_seq = max(seq_throughput)
    max_seq_idx = seq_throughput.index(max_seq)
    ax2.annotate(f'{max_seq:.1f} MB/s',
                xy=(data_sizes[max_seq_idx], max_seq),
                xytext=(10, 15), textcoords='offset points',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', fc='#FF6B6B', alpha=0.3),
                arrowprops=dict(arrowstyle='->', color='#FF6B6B', lw=2))

    # Right plot: Throughput vs Thread Count (Thread Scaling Analysis)
    # Data from CPU benchmark CSV - showing how performance scales with thread count
    thread_counts = [1, 2, 4, 8, 16]

    # Test 4: 100K Variable Size Items (27.5 MB)
    test4_throughput = [7.50, 5.01, 6.19, 6.94, 6.90]  # MB/s at each thread count

    # Test 5: 500K Medium Workload (100 MB)
    test5_throughput = [7.66, 5.10, 6.19, 6.99, 7.24]

    # Test 6: 1M Large Uniform Blocks (256 MB)
    test6_throughput = [7.65, 4.99, 6.04, 7.00, 7.12]

    ax3.plot(thread_counts, test4_throughput, 'o-', linewidth=3, markersize=12,
             label='100K items (27.5 MB)', color='#FF6B6B')
    ax3.plot(thread_counts, test5_throughput, 's-', linewidth=3, markersize=12,
             label='500K items (100 MB)', color='#4ECDC4')
    ax3.plot(thread_counts, test6_throughput, '^-', linewidth=3, markersize=12,
             label='1M items (256 MB)', color='#95E1D3')

    # Add ideal linear scaling line for reference
    ideal_scaling = [test4_throughput[0] * (1 + (t-1) * 0.9) for t in thread_counts]
    ax3.plot(thread_counts, ideal_scaling, '--', linewidth=2, alpha=0.5,
             label='Ideal Linear Scaling', color='gray')

    ax3.set_xlabel('Number of Threads', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Throughput (MB/s)', fontsize=13, fontweight='bold')
    ax3.set_title('Thread Scaling Analysis\n(Multithread Performance)',
                  fontsize=14, fontweight='bold', pad=20)
    ax3.set_xticks(thread_counts)
    ax3.legend(fontsize=11, loc='lower right')
    ax3.grid(True, alpha=0.3)

    # Annotate the non-linear scaling issue
    ax3.annotate('Poor scaling:\n2 threads slower than 1!',
                xy=(2, test4_throughput[1]),
                xytext=(-40, 20), textcoords='offset points',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.6),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    ax3.annotate('Plateau:\nNo gain beyond 8 threads',
                xy=(8, test6_throughput[3]),
                xytext=(10, -30), textcoords='offset points',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', fc='orange', alpha=0.6),
                arrowprops=dict(arrowstyle='->', color='darkorange', lw=2))

    plt.tight_layout()
    plt.savefig('plots/1_cpu_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: plots/1_cpu_comparison.png")


def plot_gpu_chunk_optimization():
    """
    Plot 2: GPU Chunk Size Optimization Study
    Shows how chunk size affects GPU performance across different dataset scales
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    datasets = [
        (chunk_opt_10k, '10K Items (2.6 MB)', 0),
        (chunk_opt_100k, '100K Items (24 MB)', 1),
        (chunk_opt_10m, '10M Items (1.8 GB)', 2)
    ]

    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']

    for idx, (data, title, dataset_idx) in enumerate(datasets):
        chunk_sizes = np.array(data['chunk_sizes'])
        color = colors[idx]

        # Top row: Execution time
        ax_time = axes[0, idx]
        ax_time.plot(chunk_sizes, data['time_ms'], 'o-', linewidth=3,
                     markersize=10, color=color, markerfacecolor='white',
                     markeredgewidth=2, markeredgecolor=color)
        ax_time.set_xlabel('Chunk Size', fontsize=12, fontweight='bold')
        ax_time.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
        ax_time.set_title(f'{title}\nExecution Time', fontsize=13, fontweight='bold')
        ax_time.set_xscale('log', base=2)
        ax_time.grid(True, alpha=0.3)

        # Mark optimal point
        min_idx = np.argmin(data['time_ms'])
        optimal_chunk = chunk_sizes[min_idx]
        optimal_time = data['time_ms'][min_idx]
        ax_time.scatter([optimal_chunk], [optimal_time], s=300,
                       facecolors='gold', edgecolors='darkgoldenrod',
                       linewidths=3, zorder=5, marker='*', label='Optimal')
        ax_time.legend(fontsize=10)

        # Annotate optimal
        ax_time.annotate(f'Optimal: {optimal_chunk//1024}K\n{optimal_time:.1f} ms',
                        xy=(optimal_chunk, optimal_time),
                        xytext=(20, 20), textcoords='offset points',
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.5', fc='gold', alpha=0.6),
                        arrowprops=dict(arrowstyle='->', lw=2))

        # Bottom row: Speedup
        ax_speedup = axes[1, idx]
        ax_speedup.plot(chunk_sizes, data['speedup'], 's-', linewidth=3,
                       markersize=10, color=color, markerfacecolor='white',
                       markeredgewidth=2, markeredgecolor=color)
        ax_speedup.set_xlabel('Chunk Size', fontsize=12, fontweight='bold')
        ax_speedup.set_ylabel('Speedup vs CPU', fontsize=12, fontweight='bold')
        ax_speedup.set_title(f'{title}\nGPU Speedup', fontsize=13, fontweight='bold')
        ax_speedup.set_xscale('log', base=2)
        ax_speedup.grid(True, alpha=0.3)
        ax_speedup.axhline(y=1, color='red', linestyle='--', alpha=0.5,
                          linewidth=2, label='No speedup')

        # Mark best speedup
        max_speedup_idx = np.argmax(data['speedup'])
        best_chunk = chunk_sizes[max_speedup_idx]
        best_speedup = data['speedup'][max_speedup_idx]
        ax_speedup.scatter([best_chunk], [best_speedup], s=300,
                          facecolors='limegreen', edgecolors='darkgreen',
                          linewidths=3, zorder=5, marker='*', label='Best Speedup')
        ax_speedup.legend(fontsize=10)

        # Annotate best
        ax_speedup.annotate(f'{best_speedup:.2f}x',
                           xy=(best_chunk, best_speedup),
                           xytext=(20, -20), textcoords='offset points',
                           fontsize=11, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.5', fc='limegreen', alpha=0.6),
                           arrowprops=dict(arrowstyle='->', lw=2))

    plt.tight_layout()
    plt.savefig('plots/2_gpu_chunk_optimization.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: plots/2_gpu_chunk_optimization.png")


def plot_4way_comparison():
    """
    Plot 3: 4-Way Implementation Comparison
    Sequential vs Multithread (16T) vs GPU Standard vs GPU Adaptive
    """
    fig = plt.figure(figsize=(18, 10))

    # Prepare data for all workloads
    workloads_small = ['Small (1K x 100B)', 'Medium (10K x 50-500B)', 'Large (100K x 256B)']
    workloads_large = ['Very Large (1M)', 'Extreme (10M)']

    # Top: Small to medium workloads (all 4 implementations)
    ax1 = plt.subplot(2, 2, 1)
    labels_small = [w.split('(')[0].strip() for w in workloads_small]
    seq_small = [workload_data[w]['Sequential'] for w in workloads_small]
    mt_small = [workload_data[w]['Multithread'] for w in workloads_small]
    gpu_small = [workload_data[w]['GPU'] for w in workloads_small]

    x = np.arange(len(labels_small))
    width = 0.2

    ax1.bar(x - 1.5*width, seq_small, width, label='Sequential', alpha=0.85, color='#FF6B6B')
    ax1.bar(x - 0.5*width, mt_small, width, label='Multithread (16T)', alpha=0.85, color='#4ECDC4')
    ax1.bar(x + 0.5*width, gpu_small, width, label='GPU Standard', alpha=0.85, color='#95E1D3')
    # No adaptive data for small datasets

    ax1.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Small to Medium Workloads\n(All Implementations)', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_small, rotation=15, ha='right', fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_yscale('log')

    # Top right: Large workloads (GPU only + adaptive)
    ax2 = plt.subplot(2, 2, 2)
    labels_large = [w.split('(')[0].strip() for w in workloads_large]
    gpu_large = [workload_data[w]['GPU'] for w in workloads_large]
    adaptive_large = [workload_data[w]['Adaptive'] for w in workloads_large]

    x2 = np.arange(len(labels_large))
    width2 = 0.35

    ax2.bar(x2 - width2/2, gpu_large, width2, label='GPU Standard', alpha=0.85, color='#95E1D3')
    ax2.bar(x2 + width2/2, adaptive_large, width2, label='GPU Adaptive', alpha=0.85, color='#F38181')

    ax2.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
    ax2.set_title('Large Workloads\n(GPU Variants)', fontsize=13, fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels_large, fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (std, adp) in enumerate(zip(gpu_large, adaptive_large)):
        ax2.text(i - width2/2, std, f'{std/1000:.1f}s',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax2.text(i + width2/2, adp, f'{adp/1000:.1f}s',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Bottom left: Throughput comparison
    ax3 = plt.subplot(2, 2, 3)

    all_workloads = workloads_small + workloads_large
    all_labels = [w.split('(')[0].strip() for w in all_workloads]
    all_sizes = [workload_data[w]['size_mb'] for w in all_workloads]

    # GPU throughput
    gpu_throughput = []
    for w in all_workloads:
        gpu_time = workload_data[w].get('GPU')
        if gpu_time:
            gpu_throughput.append(workload_data[w]['size_mb'] / (gpu_time / 1000))
        else:
            gpu_throughput.append(None)

    # Sequential throughput (where available)
    seq_throughput = []
    for w in all_workloads:
        seq_time = workload_data[w].get('Sequential')
        if seq_time:
            seq_throughput.append(workload_data[w]['size_mb'] / (seq_time / 1000))
        else:
            seq_throughput.append(None)

    # Plot only available data
    valid_seq = [(all_sizes[i], seq_throughput[i]) for i in range(len(seq_throughput))
                 if seq_throughput[i] is not None]
    valid_gpu = [(all_sizes[i], gpu_throughput[i]) for i in range(len(gpu_throughput))
                 if gpu_throughput[i] is not None]

    if valid_seq:
        seq_x, seq_y = zip(*valid_seq)
        ax3.plot(seq_x, seq_y, 'o-', linewidth=3, markersize=12,
                label='Sequential', color='#FF6B6B')

    if valid_gpu:
        gpu_x, gpu_y = zip(*valid_gpu)
        ax3.plot(gpu_x, gpu_y, 's-', linewidth=3, markersize=12,
                label='GPU Standard', color='#95E1D3')

    ax3.set_xlabel('Dataset Size (MB)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Throughput (MB/s)', fontsize=12, fontweight='bold')
    ax3.set_title('Throughput vs Dataset Size', fontsize=13, fontweight='bold')
    ax3.set_xscale('log')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)

    # Bottom right: Speedup analysis
    ax4 = plt.subplot(2, 2, 4)

    # Calculate speedups for datasets with CPU baseline
    speedup_labels = []
    gpu_speedups = []

    for w in workloads_small:
        seq_time = workload_data[w]['Sequential']
        gpu_time = workload_data[w]['GPU']
        if seq_time and gpu_time:
            speedup_labels.append(w.split('(')[0].strip())
            gpu_speedups.append(seq_time / gpu_time)

    x4 = np.arange(len(speedup_labels))
    bars = ax4.bar(x4, gpu_speedups, alpha=0.85, color='#95E1D3', edgecolor='darkslategray', linewidth=2)

    ax4.set_ylabel('Speedup vs Sequential CPU', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Workload', fontsize=12, fontweight='bold')
    ax4.set_title('GPU Speedup Over Sequential CPU', fontsize=13, fontweight='bold')
    ax4.set_xticks(x4)
    ax4.set_xticklabels(speedup_labels, rotation=15, ha='right', fontsize=11)
    ax4.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.5, label='No speedup')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        label_text = f'{height:.2f}x' if height >= 1 else f'{height:.2f}x'
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                label_text,
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig('plots/3_four_way_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: plots/3_four_way_comparison.png")


def main():
    print("\n╔════════════════════════════════════════════════════════════╗")
    print("║       GPU Merkle Tree Benchmark Visualization             ║")
    print("║       3 Plots: CPU, Chunk Optimization, 4-Way Compare     ║")
    print("╚════════════════════════════════════════════════════════════╝\n")

    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)

    plot_cpu_comparison()
    plot_gpu_chunk_optimization()
    plot_4way_comparison()

    print("\n✓ All visualizations generated successfully!")
    print("\nGenerated files in plots/ directory:")
    print("  1. plots/1_cpu_comparison.png - CPU analysis (3 subplots)")
    print("     - Execution time comparison")
    print("     - Throughput vs dataset size")
    print("     - Thread scaling analysis (1-16 threads)")
    print("  2. plots/2_gpu_chunk_optimization.png - GPU chunk size study")
    print("  3. plots/3_four_way_comparison.png - 4-way implementation comparison")
    print("\nNote: Hybrid and Streams implementations excluded as requested")


if __name__ == '__main__':
    main()

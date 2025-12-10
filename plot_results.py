#!/usr/bin/env python3
"""
GPU Merkle Tree Benchmark Visualization
Generates publication-quality plots from benchmark results
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Benchmark data extracted from the output
workload_comparison = {
    'Small (1K x 100B)': {
        'Sequential': 10.21,
        'Multithread': 11.98,
        'GPU': 71.67,
        'Hybrid': 1.72,
        'size_mb': 0.095
    },
    'Medium (10K x 50-500B)': {
        'Sequential': 271.57,
        'Multithread': 283.07,
        'GPU': 22.53,
        'Hybrid': 45.98,
        'size_mb': 2.62
    },
    'Large (100K x 256B)': {
        'Sequential': 2587.99,
        'Multithread': 2653.08,
        'GPU': 211.04,
        'Hybrid': 437.03,
        'size_mb': 24.41
    },
    'Structured (50K)': {
        'Sequential': 275.79,
        'Multithread': 283.18,
        'GPU': 103.25,
        'Hybrid': 46.67,
        'size_mb': 2.65
    },
    'Very Large (1M)': {
        'Sequential': None,  # Skipped
        'Multithread': None,  # Skipped
        'GPU': 2417.88,
        'Hybrid': 286.12,
        'size_mb': 143.05
    },
    'Extreme (10M)': {
        'Sequential': None,  # Skipped
        'Multithread': None,  # Skipped
        'GPU': 29591.08,
        'Hybrid': 3661.66,
        'size_mb': 1830.0
    }
}

# Chunk size optimization data for 10K dataset
chunk_opt_10k = {
    'chunk_sizes': [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576],
    'time_ms': [41.28, 30.27, 25.24, 22.85, 21.64, 21.10, 21.37, 21.03, 20.99, 20.99, 20.91, 21.56, 20.94, 21.01],
    'speedup': [6.72, 9.16, 10.99, 12.14, 12.82, 13.15, 12.98, 13.19, 13.21, 13.21, 13.27, 12.87, 13.25, 13.20],
    'h2d_ms': [14.70, 10.17, 8.48, 7.72, 7.31, 7.08, 7.33, 7.16, 7.14, 7.10, 7.04, 7.25, 7.03, 7.12],
    'kernel_ms': [7.49, 4.31, 2.23, 1.20, 0.71, 0.52, 0.44, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36],
    'd2h_ms': [1.04, 0.55, 0.35, 0.25, 0.20, 0.18, 0.18, 0.17, 0.16, 0.17, 0.16, 0.17, 0.17, 0.17]
}

# Chunk size optimization data for 100K dataset
chunk_opt_100k = {
    'chunk_sizes': [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576],
    'time_ms': [367.03, 291.65, 246.27, 224.16, 212.11, 206.87, 204.94, 204.30, 206.13, 207.58, 208.23, 207.75, 208.50, 208.02],
    'speedup': [7.10, 8.93, 10.57, 11.62, 12.28, 12.59, 12.71, 12.75, 12.63, 12.55, 12.51, 12.54, 12.49, 12.52],
    'h2d_ms': [136.93, 97.51, 80.12, 71.94, 67.88, 65.61, 64.26, 63.43, 64.13, 64.05, 64.26, 64.02, 64.12, 64.09],
    'kernel_ms': [41.75, 30.99, 15.85, 8.04, 4.18, 2.50, 2.05, 1.73, 2.63, 3.25, 3.60, 3.63, 3.65, 3.62],
    'd2h_ms': [9.18, 5.08, 2.94, 1.92, 1.40, 1.18, 1.03, 0.97, 0.95, 0.89, 0.87, 0.86, 0.88, 0.86]
}

# Chunk size optimization data for 10M dataset
chunk_opt_10m = {
    'chunk_sizes': [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576],
    'time_ms': [41205.70, 30793.95, 26481.06, 24482.73, 23220.53, 22473.94, 22081.72, 22082.15, 22216.39, 22369.91, 22395.59, 22311.92, 22363.82, 22374.78],
    'speedup': [5.00, 6.69, 7.78, 8.41, 8.87, 9.16, 9.32, 9.32, 9.27, 9.20, 9.19, 9.23, 9.21, 9.20],
    'h2d_ms': [13640.57, 9590.86, 7951.80, 7189.24, 6715.59, 6431.54, 6247.28, 6167.93, 6160.60, 6174.33, 6216.89, 6183.81, 6188.81, 6209.92],
    'kernel_ms': [4736.38, 2761.16, 1393.49, 707.01, 361.75, 224.67, 175.40, 140.95, 231.77, 244.57, 248.44, 226.36, 209.10, 200.90],
    'd2h_ms': [905.14, 504.19, 291.37, 188.65, 135.62, 109.39, 96.42, 89.67, 89.09, 81.55, 75.65, 71.70, 70.03, 65.51]
}

# Adaptive vs Manual comparison
adaptive_comparison = {
    'Medium (50K x 256B)': {
        'manual_ms': 102.79,
        'manual_chunk': 8192,
        'adaptive_ms': 104.52,
        'adaptive_chunk': 131072,
        'improvement_pct': -1.66
    },
    'Large (200K x 256B)': {
        'manual_ms': 415.09,
        'manual_chunk': 8192,
        'adaptive_ms': 422.17,
        'adaptive_chunk': 131072,
        'improvement_pct': -1.68
    },
    'Very Large (500K x 128B)': {
        'manual_ms': 1023.47,
        'manual_chunk': 8192,
        'adaptive_ms': 1157.81,
        'adaptive_chunk': 131072,
        'improvement_pct': -11.60
    }
}


def plot_workload_comparison():
    """Plot 1: Implementation comparison across workloads"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Filter workloads that have all implementations
    small_workloads = ['Small (1K x 100B)', 'Medium (10K x 50-500B)',
                      'Large (100K x 256B)', 'Structured (50K)']
    large_workloads = ['Very Large (1M)', 'Extreme (10M)']

    # Small to medium workloads (all implementations)
    workload_names = []
    seq_times = []
    mt_times = []
    gpu_times = []
    hybrid_times = []

    for wl in small_workloads:
        data = workload_comparison[wl]
        workload_names.append(wl.split('(')[0].strip())
        seq_times.append(data['Sequential'])
        mt_times.append(data['Multithread'])
        gpu_times.append(data['GPU'])
        hybrid_times.append(data['Hybrid'])

    x = np.arange(len(workload_names))
    width = 0.2

    ax1.bar(x - 1.5*width, seq_times, width, label='Sequential', alpha=0.8)
    ax1.bar(x - 0.5*width, mt_times, width, label='Multithread', alpha=0.8)
    ax1.bar(x + 0.5*width, gpu_times, width, label='GPU', alpha=0.8)
    ax1.bar(x + 1.5*width, hybrid_times, width, label='Hybrid', alpha=0.8)

    ax1.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Small to Medium Workloads\n(All Implementations)', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(workload_names, rotation=20, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Large workloads (GPU vs Hybrid only)
    large_names = []
    large_gpu = []
    large_hybrid = []

    for wl in large_workloads:
        data = workload_comparison[wl]
        large_names.append(wl.split('(')[0].strip())
        large_gpu.append(data['GPU'])
        large_hybrid.append(data['Hybrid'])

    x2 = np.arange(len(large_names))
    width2 = 0.35

    ax2.bar(x2 - width2/2, large_gpu, width2, label='GPU', alpha=0.8, color='C2')
    ax2.bar(x2 + width2/2, large_hybrid, width2, label='Hybrid', alpha=0.8, color='C3')

    ax2.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
    ax2.set_title('Large Workloads\n(CPU Baseline Skipped)', fontsize=13, fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(large_names)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('workload_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: workload_comparison.png")


def plot_chunk_size_optimization():
    """Plot 2: Chunk size optimization analysis"""
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    datasets = [
        (chunk_opt_10k, '10K Items (2.6 MB)', 0),
        (chunk_opt_100k, '100K Items (24 MB)', 1),
        (chunk_opt_10m, '10M Items (1.8 GB)', 2)
    ]

    for data, title, row in datasets:
        chunk_sizes = np.array(data['chunk_sizes'])

        # Plot 1: Total time
        ax1 = fig.add_subplot(gs[row, 0])
        ax1.plot(chunk_sizes, data['time_ms'], 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('Chunk Size', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Total Time (ms)', fontsize=11, fontweight='bold')
        ax1.set_title(f'{title}\nTotal Execution Time', fontsize=12, fontweight='bold')
        ax1.set_xscale('log', base=2)
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=8192, color='red', linestyle='--', alpha=0.5, label='Optimal: 8K')
        ax1.legend()

        # Plot 2: Speedup
        ax2 = fig.add_subplot(gs[row, 1])
        ax2.plot(chunk_sizes, data['speedup'], 's-', linewidth=2, markersize=8, color='green')
        ax2.set_xlabel('Chunk Size', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Speedup vs CPU', fontsize=11, fontweight='bold')
        ax2.set_title(f'{title}\nGPU Speedup', fontsize=12, fontweight='bold')
        ax2.set_xscale('log', base=2)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=max(data['speedup']), color='gold', linestyle='--', alpha=0.5,
                   label=f'Peak: {max(data["speedup"]):.2f}x')
        ax2.legend()

        # Plot 3: Time breakdown
        ax3 = fig.add_subplot(gs[row, 2])
        ax3.plot(chunk_sizes, data['h2d_ms'], '^-', linewidth=2, markersize=6, label='H2D Transfer')
        ax3.plot(chunk_sizes, data['kernel_ms'], 'v-', linewidth=2, markersize=6, label='Kernel Exec')
        ax3.plot(chunk_sizes, data['d2h_ms'], 'o-', linewidth=2, markersize=6, label='D2H Transfer')
        ax3.set_xlabel('Chunk Size', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Time (ms)', fontsize=11, fontweight='bold')
        ax3.set_title(f'{title}\nTime Breakdown', fontsize=12, fontweight='bold')
        ax3.set_xscale('log', base=2)
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

    plt.savefig('chunk_size_optimization.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: chunk_size_optimization.png")


def plot_throughput_analysis():
    """Plot 3: Throughput analysis"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Calculate throughput for each workload
    workloads = []
    data_sizes = []
    gpu_throughput = []
    hybrid_throughput = []

    for name, data in workload_comparison.items():
        if data['GPU'] is not None:
            workloads.append(name.split('(')[0].strip())
            data_sizes.append(data['size_mb'])
            gpu_throughput.append(data['size_mb'] / (data['GPU'] / 1000))  # MB/s
            hybrid_throughput.append(data['size_mb'] / (data['Hybrid'] / 1000))  # MB/s

    # Plot 1: Throughput vs data size
    ax1.plot(data_sizes, gpu_throughput, 'o-', linewidth=2, markersize=10,
             label='GPU Standard', color='steelblue')
    ax1.plot(data_sizes, hybrid_throughput, 's-', linewidth=2, markersize=10,
             label='Hybrid (simulated)', color='orange')
    ax1.set_xlabel('Dataset Size (MB)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Throughput (MB/s)', fontsize=12, fontweight='bold')
    ax1.set_title('Throughput vs Dataset Size', fontsize=13, fontweight='bold')
    ax1.set_xscale('log')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Add annotations for interesting points
    max_gpu_idx = np.argmax(gpu_throughput)
    ax1.annotate(f'Peak GPU:\n{gpu_throughput[max_gpu_idx]:.1f} MB/s',
                xy=(data_sizes[max_gpu_idx], gpu_throughput[max_gpu_idx]),
                xytext=(20, 20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    # Plot 2: Speedup comparison
    speedups = []
    speedup_labels = []

    for name, data in workload_comparison.items():
        if data['Sequential'] is not None and data['GPU'] is not None:
            speedups.append(data['Sequential'] / data['GPU'])
            speedup_labels.append(name.split('(')[0].strip())

    x = np.arange(len(speedup_labels))
    bars = ax2.bar(x, speedups, alpha=0.8, color='mediumseagreen')
    ax2.set_xlabel('Workload', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Speedup vs Sequential CPU', fontsize=12, fontweight='bold')
    ax2.set_title('GPU Speedup Over Sequential CPU', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(speedup_labels, rotation=20, ha='right')
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='No speedup')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}x',
                ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('throughput_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: throughput_analysis.png")


def plot_time_breakdown():
    """Plot 4: Detailed time breakdown for GPU operations"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    datasets = [
        (chunk_opt_10k, '10K Items', axes[0]),
        (chunk_opt_100k, '100K Items', axes[1]),
        (chunk_opt_10m, '10M Items', axes[2])
    ]

    for data, title, ax in datasets:
        # Find optimal chunk size (best speedup)
        optimal_idx = np.argmax(data['speedup'])
        chunk_sizes = data['chunk_sizes']

        # Take every 3rd chunk size for cleaner visualization
        indices = list(range(0, len(chunk_sizes), 2))
        selected_chunks = [chunk_sizes[i] for i in indices]

        h2d = [data['h2d_ms'][i] for i in indices]
        kernel = [data['kernel_ms'][i] for i in indices]
        d2h = [data['d2h_ms'][i] for i in indices]

        x = np.arange(len(selected_chunks))
        width = 0.6

        # Stacked bar chart
        p1 = ax.bar(x, h2d, width, label='H2D Transfer', alpha=0.8)
        p2 = ax.bar(x, kernel, width, bottom=h2d, label='Kernel', alpha=0.8)
        p3 = ax.bar(x, d2h, width,
                   bottom=np.array(h2d) + np.array(kernel),
                   label='D2H Transfer', alpha=0.8)

        ax.set_ylabel('Time (ms)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Chunk Size', fontsize=11, fontweight='bold')
        ax.set_title(f'{title}\nGPU Operation Breakdown', fontsize=12, fontweight='bold')
        ax.set_xticks(x[::2])  # Show every other label
        ax.set_xticklabels([f'{c//1024}K' if c >= 1024 else str(c)
                           for c in selected_chunks[::2]], rotation=45)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('time_breakdown.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: time_breakdown.png")


def plot_scaling_analysis():
    """Plot 5: Scaling analysis across data sizes"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Collect data for scaling analysis
    data_points = []
    for name, data in workload_comparison.items():
        if data['GPU'] is not None:
            data_points.append({
                'name': name,
                'size_mb': data['size_mb'],
                'gpu_time': data['GPU'],
                'seq_time': data['Sequential'],
                'mt_time': data['Multithread']
            })

    # Sort by size
    data_points.sort(key=lambda x: x['size_mb'])

    sizes = [d['size_mb'] for d in data_points]
    gpu_times = [d['gpu_time'] for d in data_points]
    seq_times = [d['seq_time'] if d['seq_time'] is not None else None for d in data_points]

    # Plot 1: Execution time vs dataset size
    ax1.plot(sizes, gpu_times, 'o-', linewidth=2, markersize=10, label='GPU', color='steelblue')

    # Add CPU times where available
    seq_sizes = [sizes[i] for i in range(len(sizes)) if seq_times[i] is not None]
    seq_vals = [t for t in seq_times if t is not None]
    if seq_vals:
        ax1.plot(seq_sizes, seq_vals, 's-', linewidth=2, markersize=10,
                label='Sequential CPU', color='coral')

    ax1.set_xlabel('Dataset Size (MB)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Execution Time Scaling', fontsize=13, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot 2: GPU efficiency (items processed per millisecond)
    items_processed = [10**3, 10**4, 10**5, 50*10**3, 10**6, 10**7]  # Approximate item counts
    efficiency = [items / gpu_times[i] for i, items in enumerate(items_processed)]

    ax2.plot(sizes, efficiency, 'o-', linewidth=2, markersize=10, color='green')
    ax2.set_xlabel('Dataset Size (MB)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Items Processed / ms', fontsize=12, fontweight='bold')
    ax2.set_title('GPU Processing Efficiency', fontsize=13, fontweight='bold')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)

    # Annotate peak efficiency
    max_eff_idx = np.argmax(efficiency)
    ax2.annotate(f'Peak Efficiency\n{efficiency[max_eff_idx]:.0f} items/ms',
                xy=(sizes[max_eff_idx], efficiency[max_eff_idx]),
                xytext=(20, -30), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'))

    plt.tight_layout()
    plt.savefig('scaling_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: scaling_analysis.png")


def plot_adaptive_comparison():
    """Plot 6: Adaptive vs Manual chunk size comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    workloads = list(adaptive_comparison.keys())
    workload_labels = [w.split('(')[0].strip() for w in workloads]

    manual_times = [adaptive_comparison[w]['manual_ms'] for w in workloads]
    adaptive_times = [adaptive_comparison[w]['adaptive_ms'] for w in workloads]
    improvements = [adaptive_comparison[w]['improvement_pct'] for w in workloads]

    x = np.arange(len(workload_labels))
    width = 0.35

    # Plot 1: Time comparison
    bars1 = ax1.bar(x - width/2, manual_times, width, label='Manual (8K chunk)', alpha=0.8)
    bars2 = ax1.bar(x + width/2, adaptive_times, width, label='Adaptive (131K chunk)', alpha=0.8)

    ax1.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Workload', fontsize=12, fontweight='bold')
    ax1.set_title('Manual vs Adaptive Chunk Size', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(workload_labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}',
                    ha='center', va='bottom', fontsize=9)

    # Plot 2: Improvement percentage
    colors = ['red' if imp < 0 else 'green' for imp in improvements]
    bars = ax2.bar(x, improvements, alpha=0.8, color=colors)

    ax2.set_ylabel('Performance Change (%)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Workload', fontsize=12, fontweight='bold')
    ax2.set_title('Adaptive Optimization Impact\n(Negative = Slower)', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(workload_labels)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom' if height > 0 else 'top',
                fontweight='bold')

    plt.tight_layout()
    plt.savefig('adaptive_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: adaptive_comparison.png")


def main():
    """Generate all plots"""
    print("\n╔════════════════════════════════════════════════════════════╗")
    print("║       GPU Merkle Tree Benchmark Visualization             ║")
    print("╚════════════════════════════════════════════════════════════╝\n")

    plot_workload_comparison()
    plot_chunk_size_optimization()
    plot_throughput_analysis()
    plot_time_breakdown()
    plot_scaling_analysis()
    plot_adaptive_comparison()

    print("\n✓ All plots generated successfully!")
    print("  Generated files:")
    print("    - workload_comparison.png")
    print("    - chunk_size_optimization.png")
    print("    - throughput_analysis.png")
    print("    - time_breakdown.png")
    print("    - scaling_analysis.png")
    print("    - adaptive_comparison.png")


if __name__ == '__main__':
    main()

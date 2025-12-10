#!/usr/bin/env python3
"""
Create an additional diagram showing WHY certain chunk sizes are optimal
Shows the three competing factors: overhead, parallelism, memory
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Create plots directory
os.makedirs('plots', exist_ok=True)

# Theoretical model of chunk size trade-offs
chunk_sizes = np.array([128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576])
chunk_labels = ['128', '256', '512', '1K', '2K', '4K', '8K', '16K', '32K', '64K', '131K', '262K', '524K', '1M']

# Three competing factors (normalized 0-1, lower is better)
# Overhead: decreases with chunk size (fewer kernel launches)
overhead = 1.0 / (np.log2(chunk_sizes) + 1)
overhead = (overhead - overhead.min()) / (overhead.max() - overhead.min())

# Parallelism efficiency: peaks at mid-range (sweet spot)
parallelism = np.exp(-((np.log2(chunk_sizes) - 13)**2) / 8)  # Peak at 8K (2^13)

# Memory pressure: increases with chunk size (more memory per transfer)
memory_pressure = (np.log2(chunk_sizes) - 7) / 14  # Normalized
memory_pressure = np.clip(memory_pressure, 0, 1)

# Combined performance (higher is better): high parallelism, low overhead, low memory
performance = parallelism * (1 - overhead * 0.3) * (1 - memory_pressure * 0.4)

# Normalize performance to 0-100 scale
performance = (performance - performance.min()) / (performance.max() - performance.min()) * 100

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Top plot: Three factors
ax1.plot(range(len(chunk_sizes)), overhead * 100, 'o-', linewidth=3, markersize=8,
         label='Kernel Launch Overhead', color='#FF6B6B')
ax1.plot(range(len(chunk_sizes)), parallelism * 100, 's-', linewidth=3, markersize=8,
         label='Parallelism Efficiency', color='#4ECDC4')
ax1.plot(range(len(chunk_sizes)), memory_pressure * 100, '^-', linewidth=3, markersize=8,
         label='Memory Pressure', color='#FFD93D')

ax1.set_xlabel('Chunk Size', fontsize=13, fontweight='bold')
ax1.set_ylabel('Factor Intensity (%)', fontsize=13, fontweight='bold')
ax1.set_title('Chunk Size Trade-offs: Three Competing Factors', fontsize=15, fontweight='bold', pad=20)
ax1.set_xticks(range(len(chunk_sizes)))
ax1.set_xticklabels(chunk_labels, rotation=45, ha='right')
ax1.legend(fontsize=11, loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 110)

# Add annotations for key regions
ax1.axvspan(-0.5, 2.5, alpha=0.15, color='red', label='High Overhead Zone')
ax1.axvspan(5.5, 8.5, alpha=0.15, color='green', label='Optimal Zone')
ax1.axvspan(10.5, 13.5, alpha=0.15, color='orange', label='Memory Bound Zone')

ax1.text(1, 95, 'Overhead\nDominates', ha='center', fontsize=10, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.5', fc='#FF6B6B', alpha=0.3))
ax1.text(7, 95, 'Sweet Spot\n(8K-16K)', ha='center', fontsize=10, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.5', fc='limegreen', alpha=0.5))
ax1.text(12, 95, 'Memory\nPlateau', ha='center', fontsize=10, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.5', fc='#FFD93D', alpha=0.3))

# Bottom plot: Resulting performance
ax2.plot(range(len(chunk_sizes)), performance, 'o-', linewidth=4, markersize=12,
         color='#9D4EDD', markerfacecolor='white', markeredgewidth=3, markeredgecolor='#9D4EDD')
ax2.fill_between(range(len(chunk_sizes)), 0, performance, alpha=0.3, color='#9D4EDD')

ax2.set_xlabel('Chunk Size', fontsize=13, fontweight='bold')
ax2.set_ylabel('Overall Performance (Normalized)', fontsize=13, fontweight='bold')
ax2.set_title('Resulting GPU Performance Profile', fontsize=15, fontweight='bold', pad=20)
ax2.set_xticks(range(len(chunk_sizes)))
ax2.set_xticklabels(chunk_labels, rotation=45, ha='right')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 110)

# Mark optimal point
optimal_idx = np.argmax(performance)
ax2.scatter([optimal_idx], [performance[optimal_idx]], s=500,
           facecolors='gold', edgecolors='darkgoldenrod',
           linewidths=4, zorder=5, marker='*')
ax2.annotate(f'Optimal: {chunk_labels[optimal_idx]}\nBest Performance',
            xy=(optimal_idx, performance[optimal_idx]),
            xytext=(20, -30), textcoords='offset points',
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.7', fc='gold', alpha=0.7),
            arrowprops=dict(arrowstyle='->', lw=3, color='darkgoldenrod'))

plt.tight_layout()
plt.savefig('plots/chunk_size_tradeoffs.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plots/chunk_size_tradeoffs.png")
print("\nThis diagram explains WHY certain chunk sizes are optimal by showing:")
print("  1. Kernel launch overhead (high for small chunks)")
print("  2. Parallelism efficiency (peaks at mid-range)")
print("  3. Memory pressure (high for large chunks)")
print("  4. Combined effect → optimal at 8K-16K range")

"""
Visualize data poisoning results
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

def visualize_results():
    """Create comprehensive visualizations"""
    
    # Load results
    results = pd.read_csv('reports/poisoning_results.csv')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Data Poisoning Impact Analysis - IRIS Dataset', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Accuracy vs Poison Rate by Attack Type
    ax1 = axes[0, 0]
    for ptype in results['poison_type'].unique():
        data = results[results['poison_type'] == ptype]
        ax1.plot(data['poison_rate'] * 100, data['test_acc'], 
                marker='o', linewidth=2, markersize=8, label=ptype)
    
    ax1.set_xlabel('Poison Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Model Accuracy vs Poison Rate', fontsize=13, fontweight='bold')
    ax1.legend(title='Attack Type')
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 1.0)
    
    # Plot 2: Accuracy Degradation
    ax2 = axes[0, 1]
    poisoned = results[results['poison_rate'] > 0]
    pivot = poisoned.pivot(index='poison_rate', columns='poison_type', values='degradation')
    pivot.plot(kind='bar', ax=ax2, width=0.8)
    ax2.set_xlabel('Poison Rate', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy Degradation', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy Degradation by Attack Type', fontsize=13, fontweight='bold')
    ax2.set_xticklabels([f'{int(x*100)}%' for x in pivot.index], rotation=0)
    ax2.legend(title='Attack Type')
    ax2.grid(alpha=0.3, axis='y')
    
    # Plot 3: Heatmap of Test Accuracy
    ax3 = axes[1, 0]
    pivot_acc = results.pivot(index='poison_type', columns='poison_rate', values='test_acc')
    sns.heatmap(pivot_acc, annot=True, fmt='.3f', cmap='RdYlGn', 
                ax=ax3, cbar_kws={'label': 'Test Accuracy'}, vmin=0, vmax=1)
    ax3.set_xlabel('Poison Rate', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Attack Type', fontsize=12, fontweight='bold')
    ax3.set_title('Test Accuracy Heatmap', fontsize=13, fontweight='bold')
    ax3.set_xticklabels([f'{int(float(x)*100)}%' for x in pivot_acc.columns])
    
    # Plot 4: Comparative Bar Chart at 50% Poison
    ax4 = axes[1, 1]
    max_poison = results[results['poison_rate'] == 0.50]
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
    bars = ax4.bar(max_poison['poison_type'], max_poison['test_acc'], 
                   color=colors, edgecolor='black', linewidth=2)
    ax4.axhline(y=0.98, color='green', linestyle='--', linewidth=2, label='Baseline (Clean)')
    ax4.set_xlabel('Attack Type', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
    ax4.set_title('Accuracy at 50% Poison Rate', fontsize=13, fontweight='bold')
    ax4.set_ylim(0, 1.0)
    ax4.legend()
    ax4.grid(alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('reports/poisoning_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Visualization saved to: reports/poisoning_analysis.png")
    plt.show()

if __name__ == "__main__":
    visualize_results()
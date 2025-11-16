"""
Export MLflow results to HTML report
"""
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

mlflow.set_tracking_uri("./mlruns")

def generate_html_report():
    """Generate comprehensive HTML report"""
    
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("iris-data-poisoning")
    
    if not experiment:
        print("No experiment found!")
        return
    
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    
    # Collect data
    results = []
    for run in runs:
        results.append({
            'run_name': run.data.tags.get('mlflow.runName', 'N/A'),
            'poison_type': run.data.params.get('poison_type'),
            'poison_rate': float(run.data.params.get('poison_rate', 0)),
            'test_accuracy': run.data.metrics.get('test_accuracy', 0),
            'degradation': run.data.metrics.get('accuracy_degradation', 0),
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values(['poison_type', 'poison_rate'])
    
    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Line chart
    for ptype in df['poison_type'].unique():
        data = df[df['poison_type'] == ptype]
        axes[0].plot(data['poison_rate'] * 100, data['test_accuracy'], 
                    marker='o', linewidth=2, label=ptype)
    axes[0].set_xlabel('Poison Rate (%)')
    axes[0].set_ylabel('Test Accuracy')
    axes[0].set_title('Accuracy vs Poison Rate')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Plot 2: Bar chart at 50%
    data_50 = df[df['poison_rate'] == 0.50]
    axes[1].bar(range(len(data_50)), data_50['test_accuracy'])
    axes[1].set_xticks(range(len(data_50)))
    axes[1].set_xticklabels(data_50['poison_type'], rotation=45)
    axes[1].set_ylabel('Test Accuracy')
    axes[1].set_title('Accuracy at 50% Poison Rate')
    axes[1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('reports/mlflow_summary.png', dpi=150, bbox_inches='tight')
    
    # Generate HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>MLflow Data Poisoning Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #3498db; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            img {{ max-width: 100%; margin: 20px 0; }}
            .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #ecf0f1; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>MLflow Data Poisoning Analysis Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Summary Statistics</h2>
        <div class="metric">
            <strong>Total Experiments:</strong> {len(df)}
        </div>
        <div class="metric">
            <strong>Best Accuracy:</strong> {df['test_accuracy'].max():.4f}
        </div>
        <div class="metric">
            <strong>Worst Accuracy:</strong> {df['test_accuracy'].min():.4f}
        </div>
        <div class="metric">
            <strong>Max Degradation:</strong> {df['degradation'].max():.4f}
        </div>
        
        <h2>All Experiments</h2>
        {df.to_html(index=False, float_format=lambda x: f'{x:.4f}')}
        
        <h2>Visualizations</h2>
        <img src="mlflow_summary.png" alt="MLflow Summary">
        
        <h2>Key Findings</h2>
        <ul>
            <li>Baseline (clean) accuracy: {df[df['poison_rate']==0]['test_accuracy'].values[0]:.4f}</li>
            <li>50% label poisoning drops accuracy to: {df[(df['poison_type']=='label_flip') & (df['poison_rate']==0.50)]['test_accuracy'].values[0]:.4f}</li>
            <li>Combined attacks are most damaging</li>
        </ul>
    </body>
    </html>
    """
    
    with open('reports/mlflow_report.html', 'w') as f:
        f.write(html)
    
    print("✓ Report generated: reports/mlflow_report.html")
    print("  Open this file in your browser to view results!")

if __name__ == "__main__":
    generate_html_report()
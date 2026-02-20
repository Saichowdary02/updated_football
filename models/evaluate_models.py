"""
Model Performance Evaluation Script
====================================
This script evaluates the performance of Logistic Regression and Random Forest models
on both training and testing data, providing metrics and visualizations.

Metrics included:
- Accuracy
- F1 Score
- Recall
- Precision
- Confusion Matrix
- ROC Curve
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import os

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_models():
    """Load the trained models and scaler"""
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    with open(os.path.join(base_path, 'logistic_regression_model.pkl'), 'rb') as f:
        lr_model = pickle.load(f)
    with open(os.path.join(base_path, 'random_forest_model.pkl'), 'rb') as f:
        rf_model = pickle.load(f)
    with open(os.path.join(base_path, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    
    return lr_model, rf_model, scaler

def load_data():
    """Load training and testing data"""
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    train_df = pd.read_csv(os.path.join(base_path, 'sample_training_data.csv'))
    test_df = pd.read_csv(os.path.join(base_path, 'sample_testing_data.csv'))
    
    # Remove any rows with '.' or invalid data
    train_df = train_df[train_df['Goals'] != '.'].dropna()
    test_df = test_df[test_df['Goals'] != '.'].dropna()
    
    # Convert to numeric
    for col in train_df.columns:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
    for col in test_df.columns:
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
    
    # Drop any remaining NaN
    train_df = train_df.dropna()
    test_df = test_df.dropna()
    
    return train_df, test_df

def prepare_features(df, scaler):
    """Prepare features and labels from dataframe"""
    feature_cols = ['Goals', 'Matches', 'Assists', 'Pass Accuracy', 'Tackles', 'Saves']
    X = df[feature_cols].values
    y = df['Selected'].values.astype(int)
    X_scaled = scaler.transform(X)
    return X_scaled, y

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate all performance metrics"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
    }
    
    # Calculate ROC AUC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    metrics['ROC AUC'] = auc(fpr, tpr)
    
    return metrics, fpr, tpr

def evaluate_model(model, X, y, model_name):
    """Evaluate a single model and return metrics"""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    metrics, fpr, tpr = calculate_metrics(y, y_pred, y_prob)
    cm = confusion_matrix(y, y_pred)
    
    return {
        'predictions': y_pred,
        'probabilities': y_prob,
        'metrics': metrics,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr
    }

def create_metrics_comparison_chart(results, save_path):
    """Create a bar chart comparing metrics across models and datasets"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Performance Metrics Comparison', fontsize=16, fontweight='bold')
    
    metrics_to_plot = ['Accuracy', 'F1 Score', 'Recall', 'Precision']
    colors = {'Logistic Regression': '#4361ee', 'Random Forest': '#f72585'}
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        
        # Training data
        train_lr = results['train']['Logistic Regression']['metrics'][metric]
        train_rf = results['train']['Random Forest']['metrics'][metric]
        
        # Test data
        test_lr = results['test']['Logistic Regression']['metrics'][metric]
        test_rf = results['test']['Random Forest']['metrics'][metric]
        
        x = np.arange(2)
        width = 0.35
        
        bars1 = ax.bar(x - width/2, [train_lr, test_lr], width, 
                       label='Logistic Regression', color=colors['Logistic Regression'], alpha=0.8)
        bars2 = ax.bar(x + width/2, [train_rf, test_rf], width, 
                       label='Random Forest', color=colors['Random Forest'], alpha=0.8)
        
        ax.set_ylabel('Score')
        ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Training Data', 'Test Data'])
        ax.legend(loc='lower right')
        ax.set_ylim(0, 1.1)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Metrics comparison chart saved to: {save_path}")

def create_confusion_matrices(results, save_path):
    """Create confusion matrix visualizations"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Confusion Matrices', fontsize=16, fontweight='bold')
    
    datasets = [('train', 'Training Data'), ('test', 'Test Data')]
    models = ['Logistic Regression', 'Random Forest']
    
    for row, (data_key, data_name) in enumerate(datasets):
        for col, model_name in enumerate(models):
            ax = axes[row, col]
            cm = results[data_key][model_name]['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Not Selected', 'Selected'],
                       yticklabels=['Not Selected', 'Selected'])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title(f'{model_name}\n({data_name})', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrices saved to: {save_path}")

def create_roc_curves(results, save_path):
    """Create ROC curve comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('ROC Curves Comparison', fontsize=16, fontweight='bold')
    
    colors = {'Logistic Regression': '#4361ee', 'Random Forest': '#f72585'}
    datasets = [('train', 'Training Data'), ('test', 'Test Data')]
    
    for idx, (data_key, data_name) in enumerate(datasets):
        ax = axes[idx]
        
        for model_name in ['Logistic Regression', 'Random Forest']:
            fpr = results[data_key][model_name]['fpr']
            tpr = results[data_key][model_name]['tpr']
            roc_auc = results[data_key][model_name]['metrics']['ROC AUC']
            
            ax.plot(fpr, tpr, color=colors[model_name], lw=2,
                   label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - {data_name}', fontsize=12)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ ROC curves saved to: {save_path}")

def create_summary_table(results, save_path):
    """Create and save a summary table of all metrics"""
    summary_data = []
    
    for data_type in ['train', 'test']:
        for model_name in ['Logistic Regression', 'Random Forest']:
            metrics = results[data_type][model_name]['metrics']
            row = {
                'Dataset': 'Training' if data_type == 'train' else 'Test',
                'Model': model_name,
                **metrics
            }
            summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    
    # Round all numeric columns
    numeric_cols = ['Accuracy', 'F1 Score', 'Recall', 'Precision', 'ROC AUC']
    for col in numeric_cols:
        df[col] = df[col].apply(lambda x: f'{x:.4f}')
    
    # Save to CSV
    df.to_csv(save_path, index=False)
    
    return df

def print_detailed_report(results):
    """Print a detailed report to console"""
    print("\n" + "="*80)
    print(" MODEL PERFORMANCE EVALUATION REPORT")
    print("="*80)
    
    for data_type, data_name in [('train', 'TRAINING DATA'), ('test', 'TEST DATA')]:
        print(f"\n{'─'*80}")
        print(f" 📊 {data_name}")
        print(f"{'─'*80}")
        
        for model_name in ['Logistic Regression', 'Random Forest']:
            metrics = results[data_type][model_name]['metrics']
            print(f"\n  🤖 {model_name}")
            print(f"  {'─'*40}")
            print(f"     Accuracy:   {metrics['Accuracy']:.4f}  ({metrics['Accuracy']*100:.2f}%)")
            print(f"     F1 Score:   {metrics['F1 Score']:.4f}")
            print(f"     Recall:     {metrics['Recall']:.4f}")
            print(f"     Precision:  {metrics['Precision']:.4f}")
            print(f"     ROC AUC:    {metrics['ROC AUC']:.4f}")
    
    print("\n" + "="*80)

def main():
    """Main function to run the evaluation"""
    print("\n🚀 Starting Model Performance Evaluation...")
    print("─" * 50)
    
    # Get the base path for saving outputs
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # Load models and data
    print("\n📂 Loading models...")
    lr_model, rf_model, scaler = load_models()
    print("   ✓ Logistic Regression model loaded")
    print("   ✓ Random Forest model loaded")
    print("   ✓ Scaler loaded")
    
    print("\n📂 Loading data...")
    train_df, test_df = load_data()
    print(f"   ✓ Training data: {len(train_df)} samples")
    print(f"   ✓ Test data: {len(test_df)} samples")
    
    # Prepare features
    print("\n🔧 Preparing features...")
    X_train, y_train = prepare_features(train_df, scaler)
    X_test, y_test = prepare_features(test_df, scaler)
    
    # Evaluate models
    print("\n📈 Evaluating models...")
    results = {
        'train': {
            'Logistic Regression': evaluate_model(lr_model, X_train, y_train, 'LR'),
            'Random Forest': evaluate_model(rf_model, X_train, y_train, 'RF')
        },
        'test': {
            'Logistic Regression': evaluate_model(lr_model, X_test, y_test, 'LR'),
            'Random Forest': evaluate_model(rf_model, X_test, y_test, 'RF')
        }
    }
    
    # Print detailed report
    print_detailed_report(results)
    
    # Create and save summary table
    print("\n💾 Saving results...")
    summary_df = create_summary_table(results, os.path.join(base_path, 'model_metrics_summary.csv'))
    print("\n📋 Metrics Summary Table:")
    print(summary_df.to_string(index=False))
    print(f"\n✅ Summary table saved to: {os.path.join(base_path, 'model_metrics_summary.csv')}")
    
    # Create visualizations
    print("\n📊 Generating visualizations...")
    
    create_metrics_comparison_chart(results, os.path.join(base_path, 'metrics_comparison.png'))
    create_confusion_matrices(results, os.path.join(base_path, 'confusion_matrices.png'))
    create_roc_curves(results, os.path.join(base_path, 'roc_curves.png'))
    
    print("\n" + "="*80)
    print(" ✅ EVALUATION COMPLETE!")
    print("="*80)
    print("\n📁 Output files generated in the 'models' folder:")
    print("   • model_metrics_summary.csv - Detailed metrics table")
    print("   • metrics_comparison.png - Bar chart comparison")
    print("   • confusion_matrices.png - Confusion matrix heatmaps")
    print("   • roc_curves.png - ROC curve plots")
    print("")

if __name__ == "__main__":
    main()

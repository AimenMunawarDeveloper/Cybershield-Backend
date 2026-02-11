"""
Comprehensive visualization module for fusion model training.
Generates publication-quality plots for 2025-2026 research papers.

Includes:
- Training/validation curves (loss, accuracy)
- ROC and Precision-Recall curves
- Confusion matrix heatmaps
- Calibration plots (ECE)
- Learning rate schedule
- Per-class metrics
- Robustness analysis
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, 
    confusion_matrix, classification_report
)
import json

# Set publication-quality style
try:
    plt.style.use('seaborn-v0_8-paper')
except OSError:
    try:
        plt.style.use('seaborn-paper')
    except OSError:
        plt.style.use('seaborn')  # Fallback to basic seaborn style
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})


def plot_training_curves(
    history: Dict[str, List[float]],
    output_path: Path,
    title: str = "Training Progress"
) -> None:
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        history: Dictionary with keys 'train_loss', 'train_acc', 'val_loss', 'val_acc'
        output_path: Path to save the plot
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curve
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2, alpha=0.8)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2, alpha=0.8)
    axes[0].set_xlabel('Epoch', fontweight='bold')
    axes[0].set_ylabel('Loss', fontweight='bold')
    axes[0].set_title('Training and Validation Loss', fontweight='bold')
    axes[0].legend(loc='best', frameon=True, fancybox=True, shadow=True)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    
    # Accuracy curve
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2, alpha=0.8)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2, alpha=0.8)
    axes[1].set_xlabel('Epoch', fontweight='bold')
    axes[1].set_ylabel('Accuracy', fontweight='bold')
    axes[1].set_title('Training and Validation Accuracy', fontweight='bold')
    axes[1].legend(loc='best', frameon=True, fancybox=True, shadow=True)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle(title, fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to: {output_path}")


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    output_path: Path,
    title: str = "ROC Curve"
) -> float:
    """
    Plot ROC curve and return AUC score.
    
    Args:
        y_true: True labels
        y_pred_probs: Predicted probabilities for positive class
        output_path: Path to save the plot
        title: Plot title
        
    Returns:
        AUC score
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontweight='bold', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontweight='bold', fontsize=14)
    ax.set_title(title, fontweight='bold', fontsize=16)
    ax.legend(loc="lower right", frameon=True, fancybox=True, shadow=True, fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curve to: {output_path} (AUC = {roc_auc:.4f})")
    
    return roc_auc


def plot_pr_curve(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    output_path: Path,
    title: str = "Precision-Recall Curve"
) -> float:
    """
    Plot Precision-Recall curve and return PR-AUC score.
    
    Args:
        y_true: True labels
        y_pred_probs: Predicted probabilities for positive class
        output_path: Path to save the plot
        title: Plot title
        
    Returns:
        PR-AUC score
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_probs)
    pr_auc = auc(recall, precision)
    
    # Calculate baseline (random classifier)
    baseline = np.sum(y_true) / len(y_true)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.plot(recall, precision, color='darkblue', lw=3, label=f'PR curve (AUC = {pr_auc:.4f})')
    ax.axhline(y=baseline, color='navy', lw=2, linestyle='--', label=f'Baseline (AP = {baseline:.4f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontweight='bold', fontsize=14)
    ax.set_ylabel('Precision', fontweight='bold', fontsize=14)
    ax.set_title(title, fontweight='bold', fontsize=16)
    ax.legend(loc="lower left", frameon=True, fancybox=True, shadow=True, fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved PR curve to: {output_path} (PR-AUC = {pr_auc:.4f})")
    
    return pr_auc


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    title: str = "Confusion Matrix",
    class_names: Optional[List[str]] = None
) -> None:
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_path: Path to save the plot
        title: Plot title
        class_names: Optional class names (default: ['Legitimate', 'Phishing'])
    """
    if class_names is None:
        class_names = ['Legitimate', 'Phishing']
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, linewidths=0.5, linecolor='gray')
    axes[0].set_xlabel('Predicted Label', fontweight='bold', fontsize=14)
    axes[0].set_ylabel('True Label', fontweight='bold', fontsize=14)
    axes[0].set_title('Confusion Matrix (Counts)', fontweight='bold', fontsize=14)
    
    # Normalized percentages
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=axes[1],
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage'}, linewidths=0.5, linecolor='gray')
    axes[1].set_xlabel('Predicted Label', fontweight='bold', fontsize=14)
    axes[1].set_ylabel('True Label', fontweight='bold', fontsize=14)
    axes[1].set_title('Confusion Matrix (Normalized)', fontweight='bold', fontsize=14)
    
    plt.suptitle(title, fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to: {output_path}")


def plot_calibration_curve(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    ece: float,
    output_path: Path,
    n_bins: int = 10,
    title: str = "Calibration Curve"
) -> None:
    """
    Plot calibration curve (reliability diagram) showing model calibration.
    
    Args:
        y_true: True labels
        y_pred_probs: Predicted probabilities for positive class
        ece: Expected Calibration Error
        output_path: Path to save the plot
        n_bins: Number of bins for calibration
        title: Plot title
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_pred_probs > bin_lower) & (y_pred_probs <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_pred_probs[in_bin].mean()
            bin_accuracies.append(accuracy_in_bin)
            bin_confidences.append(avg_confidence_in_bin)
            bin_counts.append(prop_in_bin * len(y_true))
        else:
            bin_accuracies.append(0)
            bin_confidences.append((bin_lower + bin_upper) / 2)
            bin_counts.append(0)
    
    bin_accuracies = np.array(bin_accuracies)
    bin_confidences = np.array(bin_confidences)
    bin_counts = np.array(bin_counts)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot calibration curve
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
    ax.plot(bin_confidences, bin_accuracies, 'o-', color='darkblue', 
            label=f'Model (ECE = {ece:.4f})', linewidth=2, markersize=8)
    
    # Add bar chart showing sample distribution
    ax2 = ax.twinx()
    ax2.bar(bin_confidences, bin_counts, width=0.05, alpha=0.3, color='gray', label='Sample Count')
    ax2.set_ylabel('Number of Samples', fontweight='bold', fontsize=12)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Mean Predicted Probability', fontweight='bold', fontsize=14)
    ax.set_ylabel('Fraction of Positives', fontweight='bold', fontsize=14)
    ax.set_title(title, fontweight='bold', fontsize=16)
    ax.legend(loc="upper left", frameon=True, fancybox=True, shadow=True, fontsize=11)
    ax2.legend(loc="upper right", frameon=True, fancybox=True, shadow=True, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved calibration curve to: {output_path} (ECE = {ece:.4f})")


def plot_learning_rate_schedule(
    lr_history: List[float],
    output_path: Path,
    title: str = "Learning Rate Schedule"
) -> None:
    """
    Plot learning rate schedule over training.
    
    Args:
        lr_history: List of learning rates per epoch
        output_path: Path to save the plot
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(lr_history) + 1)
    ax.plot(epochs, lr_history, 'b-', linewidth=2, marker='o', markersize=4, alpha=0.8)
    
    ax.set_xlabel('Epoch', fontweight='bold', fontsize=14)
    ax.set_ylabel('Learning Rate', fontweight='bold', fontsize=14)
    ax.set_title(title, fontweight='bold', fontsize=16)
    ax.set_yscale('log')  # Log scale for better visualization
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved learning rate schedule to: {output_path}")


def plot_metrics_comparison(
    metrics: Dict[str, float],
    output_path: Path,
    title: str = "Model Performance Metrics"
) -> None:
    """
    Plot bar chart comparing different metrics.
    
    Args:
        metrics: Dictionary of metric names and values
        output_path: Path to save the plot
        title: Plot title
    """
    # Filter out non-numeric metrics
    plot_metrics = {k: v for k, v in metrics.items() 
                   if isinstance(v, (int, float)) and not k.startswith('true_') and not k.startswith('false_')}
    
    # Exclude ECE from bar chart (it's better in calibration plot)
    plot_metrics = {k: v for k, v in plot_metrics.items() if k != 'ece'}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metric_names = list(plot_metrics.keys())
    metric_values = list(plot_metrics.values())
    
    bars = ax.barh(metric_names, metric_values, color=sns.color_palette("husl", len(metric_names)))
    
    # Add value labels on bars
    for i, (name, value) in enumerate(zip(metric_names, metric_values)):
        ax.text(value + 0.01, i, f'{value:.4f}', va='center', fontweight='bold', fontsize=11)
    
    ax.set_xlabel('Score', fontweight='bold', fontsize=14)
    ax.set_ylabel('Metric', fontweight='bold', fontsize=14)
    ax.set_title(title, fontweight='bold', fontsize=16)
    ax.set_xlim([0, max(metric_values) * 1.15])
    ax.grid(True, alpha=0.3, linestyle='--', axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved metrics comparison to: {output_path}")


def plot_robustness_analysis(
    robustness_results: Dict[str, Dict[str, float]],
    output_path: Path,
    title: str = "Robustness Analysis"
) -> None:
    """
    Plot robustness analysis results (accuracy vs noise level).
    
    Args:
        robustness_results: Dictionary with noise levels as keys and metrics as values
        output_path: Path to save the plot
        title: Plot title
    """
    noise_levels = []
    accuracies = []
    stds = []
    
    for key, value in robustness_results.items():
        if key == 'baseline':
            noise_levels.append(0.0)
        else:
            noise_levels.append(float(key.replace('noise_', '')))
        accuracies.append(value['accuracy'])
        stds.append(value.get('std', 0.0))
    
    # Sort by noise level
    sorted_indices = np.argsort(noise_levels)
    noise_levels = [noise_levels[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]
    stds = [stds[i] for i in sorted_indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.errorbar(noise_levels, accuracies, yerr=stds, fmt='o-', linewidth=2, 
                markersize=8, capsize=5, capthick=2, color='darkblue', alpha=0.8)
    
    ax.set_xlabel('Noise Level', fontweight='bold', fontsize=14)
    ax.set_ylabel('Accuracy', fontweight='bold', fontsize=14)
    ax.set_title(title, fontweight='bold', fontsize=16)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved robustness analysis to: {output_path}")


def plot_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    class_names: Optional[List[str]] = None,
    title: str = "Per-Class Performance Metrics"
) -> None:
    """
    Plot per-class metrics (precision, recall, F1).
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_path: Path to save the plot
        class_names: Optional class names
        title: Plot title
    """
    if class_names is None:
        class_names = ['Legitimate', 'Phishing']
    
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, zero_division=0
    )
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.8, color='#3498db')
    bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.8, color='#2ecc71')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8, color='#e74c3c')
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax.set_xlabel('Class', fontweight='bold', fontsize=14)
    ax.set_ylabel('Score', fontweight='bold', fontsize=14)
    ax.set_title(title, fontweight='bold', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved per-class metrics to: {output_path}")


def generate_all_visualizations(
    history: Dict[str, List[float]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_probs: np.ndarray,
    metrics: Dict[str, float],
    output_dir: Path,
    lr_history: Optional[List[float]] = None,
    robustness_results: Optional[Dict[str, Dict[str, float]]] = None,
    model_name: str = "Fusion Model"
) -> Dict[str, str]:
    """
    Generate all visualizations and save them to output directory.
    
    Args:
        history: Training history dictionary
        y_true: True labels
        y_pred: Predicted labels
        y_pred_probs: Predicted probabilities
        metrics: Dictionary of computed metrics
        output_dir: Directory to save all plots
        lr_history: Optional learning rate history
        robustness_results: Optional robustness analysis results
        model_name: Name of the model for titles
        
    Returns:
        Dictionary mapping plot names to file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_paths = {}
    
    print("\n" + "=" * 70)
    print("GENERATING PUBLICATION-QUALITY VISUALIZATIONS")
    print("=" * 70)
    
    # 1. Training curves
    plot_paths['training_curves'] = str(output_dir / "training_curves.png")
    plot_training_curves(history, Path(plot_paths['training_curves']), 
                        title=f"{model_name} - Training Progress")
    
    # 2. ROC curve
    plot_paths['roc_curve'] = str(output_dir / "roc_curve.png")
    roc_auc = plot_roc_curve(y_true, y_pred_probs, Path(plot_paths['roc_curve']),
                            title=f"{model_name} - ROC Curve")
    
    # 3. Precision-Recall curve
    plot_paths['pr_curve'] = str(output_dir / "pr_curve.png")
    pr_auc = plot_pr_curve(y_true, y_pred_probs, Path(plot_paths['pr_curve']),
                           title=f"{model_name} - Precision-Recall Curve")
    
    # 4. Confusion matrix
    plot_paths['confusion_matrix'] = str(output_dir / "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, Path(plot_paths['confusion_matrix']),
                         title=f"{model_name} - Confusion Matrix")
    
    # 5. Calibration curve
    if 'ece' in metrics:
        plot_paths['calibration_curve'] = str(output_dir / "calibration_curve.png")
        plot_calibration_curve(y_true, y_pred_probs, metrics['ece'],
                              Path(plot_paths['calibration_curve']),
                              title=f"{model_name} - Calibration Curve")
    
    # 6. Learning rate schedule
    if lr_history:
        plot_paths['lr_schedule'] = str(output_dir / "learning_rate_schedule.png")
        plot_learning_rate_schedule(lr_history, Path(plot_paths['lr_schedule']),
                                   title=f"{model_name} - Learning Rate Schedule")
    
    # 7. Metrics comparison
    plot_paths['metrics_comparison'] = str(output_dir / "metrics_comparison.png")
    plot_metrics_comparison(metrics, Path(plot_paths['metrics_comparison']),
                          title=f"{model_name} - Performance Metrics")
    
    # 8. Per-class metrics
    plot_paths['per_class_metrics'] = str(output_dir / "per_class_metrics.png")
    plot_per_class_metrics(y_true, y_pred, Path(plot_paths['per_class_metrics']),
                          title=f"{model_name} - Per-Class Metrics")
    
    # 9. Robustness analysis
    if robustness_results:
        plot_paths['robustness'] = str(output_dir / "robustness_analysis.png")
        plot_robustness_analysis(robustness_results, Path(plot_paths['robustness']),
                               title=f"{model_name} - Robustness Analysis")
    
    # Save metrics to JSON
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    plot_paths['metrics_json'] = str(metrics_path)
    
    print("\n" + "=" * 70)
    print(f"All visualizations saved to: {output_dir}")
    print("=" * 70)
    
    return plot_paths

# bloom_taxonomy_evaluation/utils/metrics.py

"""
Evaluation metrics for Bloom taxonomy classification.
Using the 4 specified metrics: Accuracy, Cohen's Kappa, Krippendorff's Alpha, ICC
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, accuracy_score, classification_report
from scipy import stats
import krippendorff
from typing import Dict, List, Tuple

def calculate_agreement_metrics(human_labels: List[str], 
                               model_labels: List[str], 
                               categories: List[str]) -> Dict[str, float]:
    """
    Calculate the 4 specified metrics for Bloom taxonomy classification.
    
    Args:
        human_labels: List of human-assigned categories (ground truth)
        model_labels: List of model-assigned categories  
        categories: List of all possible categories
    
    Returns:
        Dictionary containing the 4 metrics
    """
    
    # Convert to numpy arrays
    human_array = np.array(human_labels)
    model_array = np.array(model_labels)
    
    # 1. ACCURACY - Exact match accuracy
    accuracy = accuracy_score(human_array, model_array) * 100
    
    # 2. COHEN'S KAPPA (κ) - Agreement beyond chance
    kappa = cohen_kappa_score(human_array, model_array)
    
    # 3. KRIPPENDORFF'S ALPHA (α) - Reliability measure
    # Prepare data for Krippendorff's alpha (need to convert categories to numeric)
    category_to_num = {cat: i for i, cat in enumerate(categories)}
    human_numeric = [category_to_num[label] for label in human_labels]
    model_numeric = [category_to_num[label] for label in model_labels]
    
    data = np.array([human_numeric, model_numeric])
    alpha = krippendorff.alpha(data, level_of_measurement='nominal')
    
    # 4. INTRACLASS CORRELATION (ICC) - Correlation between scores
    # For categorical data, we use the correlation between numeric representations
    correlation = np.corrcoef(human_numeric, model_numeric)[0, 1]
    icc = correlation  # For categorical data, ICC approximates to correlation
    
    # Per-category metrics
    category_metrics = {}
    
    for category in categories:
        # Binary classification for this category vs all others
        human_binary = (human_array == category).astype(int)
        model_binary = (model_array == category).astype(int)
        
        if len(np.unique(human_binary)) > 1 or len(np.unique(model_binary)) > 1:
            cat_accuracy = accuracy_score(human_binary, model_binary) * 100
            
            if len(np.unique(human_binary)) > 1 and len(np.unique(model_binary)) > 1:
                cat_kappa = cohen_kappa_score(human_binary, model_binary)
                cat_corr = np.corrcoef(human_binary, model_binary)[0, 1]
            else:
                cat_kappa = 0.0
                cat_corr = 0.0
                
            # Category-specific Krippendorff's alpha
            cat_data = np.array([human_binary, model_binary])
            cat_alpha = krippendorff.alpha(cat_data, level_of_measurement='nominal')
            
        else:
            cat_accuracy = 100.0 if np.all(human_binary == model_binary) else 0.0
            cat_kappa = 0.0
            cat_alpha = 0.0
            cat_corr = 0.0
        
        # Count true positives, false positives, etc.
        tp = np.sum((human_binary == 1) & (model_binary == 1))
        fp = np.sum((human_binary == 0) & (model_binary == 1))
        fn = np.sum((human_binary == 1) & (model_binary == 0))
        tn = np.sum((human_binary == 0) & (model_binary == 0))
        
        # Precision, Recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        category_metrics[category] = {
            'accuracy': cat_accuracy,
            'kappa': cat_kappa,
            'alpha': cat_alpha if not np.isnan(cat_alpha) else 0.0,
            'correlation': cat_corr if not np.isnan(cat_corr) else 0.0,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': np.sum(human_binary)
        }
    
    return {
        'accuracy': accuracy,
        'kappa': kappa,
        'alpha': alpha if not np.isnan(alpha) else 0.0,
        'icc': icc if not np.isnan(icc) else 0.0,
        'category_metrics': category_metrics
    }


def plot_category_performance(metrics: Dict[str, float], 
                            categories: List[str], 
                            technique_name: str, 
                            save_path: str = None):
    """
    Create visualization of per-category performance using the 4 metrics.
    """
    category_metrics = metrics['category_metrics']
    
    # Prepare data for plotting
    metric_names = ['Accuracy', 'Cohen\'s κ', 'Krippendorff\'s α', 'F1 Score']
    metric_keys = ['accuracy', 'kappa', 'alpha', 'f1']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (metric_name, metric_key) in enumerate(zip(metric_names, metric_keys)):
        scores = []
        for cat in categories:
            if metric_key == 'accuracy':
                scores.append(category_metrics[cat][metric_key] / 100)  # Convert to 0-1 scale
            else:
                scores.append(category_metrics[cat][metric_key])
        
        bars = axes[i].bar(categories, scores, alpha=0.7, 
                          color=plt.cm.Set3(np.arange(len(categories))))
        axes[i].set_title(f'{metric_name} by Category', fontsize=12, fontweight='bold')
        axes[i].set_ylabel(metric_name)
        axes[i].set_ylim(-0.1, 1.1)
        axes[i].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            if metric_key == 'accuracy':
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{score*100:.1f}%', ha='center', va='bottom', fontsize=9)
            else:
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{score:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle(f'Bloom Taxonomy Classification Performance: {technique_name}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def print_detailed_results(metrics: Dict[str, float], 
                         categories: List[str], 
                         technique_name: str):
    """
    Print detailed results summary with the 4 metrics.
    """
    print(f"\n=== {technique_name} Results ===")
    print(f"Overall Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.1f}%")
    print(f"  Cohen's Kappa (κ): {metrics['kappa']:.3f}")
    print(f"  Krippendorff's Alpha (α): {metrics['alpha']:.3f}")
    print(f"  Intraclass Correlation (ICC): {metrics['icc']:.3f}")
    
    print("\nPer-Category Results:")
    category_metrics = metrics['category_metrics']
    for category in categories:
        cat_metrics = category_metrics[category]
        print(f"  {category:12s}: Accuracy={cat_metrics['accuracy']:.1f}%, "
              f"κ={cat_metrics['kappa']:.3f}, "
              f"F1={cat_metrics['f1']:.3f}, "
              f"Support={cat_metrics['support']}")
    
    # Overall interpretation based on Kappa
    kappa = metrics['kappa']
    print(f"\nOverall Agreement Level (κ={kappa:.3f}): ", end="")
    if kappa > 0.8:
        print("Almost Perfect Agreement")
    elif kappa > 0.6:
        print("Substantial Agreement")  
    elif kappa > 0.4:
        print("Moderate Agreement")
    elif kappa > 0.2:
        print("Fair Agreement")
    elif kappa > 0:
        print("Slight Agreement")
    else:
        print("Poor Agreement")


def create_confusion_matrix(human_labels: List[str], 
                           model_labels: List[str], 
                           categories: List[str],
                           save_path: str = None):
    """
    Create and optionally save a confusion matrix for Bloom taxonomy classification.
    """
    from sklearn.metrics import confusion_matrix
    import pandas as pd
    
    cm = confusion_matrix(human_labels, model_labels, labels=categories)
    
    # Create DataFrame for better visualization
    cm_df = pd.DataFrame(cm, index=categories, columns=categories)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', 
                xticklabels=categories, yticklabels=categories)
    plt.title('Confusion Matrix: Human vs Model Classifications')
    plt.xlabel('Model Predictions')
    plt.ylabel('Human Ground Truth')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()
    
    return cm_df
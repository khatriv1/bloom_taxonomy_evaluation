# bloom_taxonomy_evaluation/evaluation/evaluate_zero_shot.py

import sys
import os
import pandas as pd
import openai
import time
from typing import Dict, List

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompting.zero_shot import get_zero_shot_prediction_binary_class, get_zero_shot_prediction_single_category
from utils.data_loader import load_and_preprocess_bloom_data, get_outcome_category, get_outcome_binary_labels, filter_valid_outcomes
from utils.metrics import calculate_agreement_metrics, plot_category_performance, print_detailed_results

def evaluate_zero_shot(data_path: str, api_key: str, output_dir: str = "results/zero_shot", limit: int = None):
    """Evaluate Zero-shot prompting technique on Bloom taxonomy dataset with binary predictions."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess data
    try:
        df = load_and_preprocess_bloom_data(data_path)
        if df.empty:
            raise Exception("No valid examples found in the data file")
        
        # Filter to valid outcomes (at least 1 label)
        df = filter_valid_outcomes(df, min_labels=1)
        
        if limit:
            df = df.head(limit)
        print(f"\nEvaluating on {len(df)} learning outcomes")
        
    except Exception as e:
        raise Exception(f"Error loading or processing data: {str(e)}")
    
    # Initialize OpenAI client
    client = openai.OpenAI(api_key=api_key)
    
    # Bloom taxonomy categories
    categories = [
        'remember', 'understand', 'apply', 'analyze', 'evaluate', 'create'
    ]
    
    # Store results
    human_labels = []  # Ground truth categories (single best)
    model_labels = []  # Model predictions (single best)
    detailed_results = []
    
    # Store binary predictions and ground truth
    all_model_binary = []  # Binary model predictions
    all_human_binary = []  # Binary ground truth labels
    
    # Process each learning outcome
    total = len(df)
    for seq, (_, row) in enumerate(df.iterrows(), start=1):
        print(f"\nProcessing outcome {seq}/{total}")
        print(f"Learning outcome: {row['learning_outcome'][:100]}...")
        
        outcome_id = str(row['outcome_id'])
        
        try:
            # Get human ground truth (primary category)
            human_category = get_outcome_category(row, method='first_selected')
            human_labels.append(human_category)
            
            # Get human binary labels for all categories
            human_binary = get_outcome_binary_labels(row)
            all_human_binary.append(human_binary)
            
            # Get model predictions with Zero-shot (binary)
            model_binary = get_zero_shot_prediction_binary_class(
                learning_outcome=row['learning_outcome'],
                client=client
            )
            all_model_binary.append(model_binary)
            
            # Get single best category for existing metrics
            model_category = get_zero_shot_prediction_single_category(
                learning_outcome=row['learning_outcome'],
                client=client
            )
            model_labels.append(model_category)
            
            # Calculate binary accuracy for this outcome
            binary_matches = sum(1 for cat in categories if human_binary[cat] == model_binary[cat])
            binary_accuracy = binary_matches / len(categories)
            
            # Store detailed result
            detailed_results.append({
                'outcome_id': outcome_id,
                'learning_outcome': row['learning_outcome'],
                'human_category': human_category,
                'model_category': model_category,
                'exact_match': human_category == model_category,
                'human_binary': human_binary,
                'model_binary': model_binary,
                'binary_accuracy': binary_accuracy,
                'binary_matches': binary_matches
            })
            
            print(f"Human (primary): {human_category}")
            print(f"Model (primary): {model_category}")
            print(f"Primary match: {human_category == model_category}")
            print(f"Human binary: {human_binary}")
            print(f"Model binary: {model_binary}")
            print(f"Binary accuracy: {binary_accuracy:.3f} ({binary_matches}/6)")
            
        except Exception as e:
            print(f"Error processing outcome {outcome_id}: {str(e)}")
            continue
        
        time.sleep(1)  # Rate limiting
    
    if not human_labels:
        raise Exception("No valid predictions were generated")
    
    # Calculate metrics using existing approach (primary category comparison)
    metrics = calculate_agreement_metrics(human_labels, model_labels, categories)
    
    # Calculate binary-level metrics
    total_binary_matches = sum(result['binary_matches'] for result in detailed_results)
    total_binary_comparisons = len(detailed_results) * len(categories)
    overall_binary_accuracy = total_binary_matches / total_binary_comparisons
    
    metrics['binary_accuracy'] = overall_binary_accuracy * 100
    print(f"\nOverall Binary Accuracy: {overall_binary_accuracy:.3f} ({total_binary_matches}/{total_binary_comparisons})")
    
    # Create visualization
    plot_category_performance(
        metrics, 
        categories, 
        'Zero-shot (Binary)',
        f"{output_dir}/zero_shot_performance.png"
    )
    
    # Print results
    print_detailed_results(metrics, categories, 'Zero-shot (Binary)')
    
    # Save detailed results including binary predictions
    results_df = pd.DataFrame(detailed_results)
    results_df.to_csv(f"{output_dir}/detailed_results.csv", index=False)
    print(f"\nDetailed results saved to {output_dir}/detailed_results.csv")
    
    # Save binary predictions in separate file for analysis
    binary_results = []
    for i, (human_binary, model_binary) in enumerate(zip(all_human_binary, all_model_binary)):
        result_row = {'outcome_id': detailed_results[i]['outcome_id']}
        # Add human binary labels
        for cat in categories:
            result_row[f'human_{cat}'] = human_binary[cat]
        # Add model binary labels
        for cat in categories:
            result_row[f'model_{cat}'] = model_binary[cat]
        binary_results.append(result_row)
    
    binary_df = pd.DataFrame(binary_results)
    binary_df.to_csv(f"{output_dir}/binary_predictions.csv", index=False)
    print(f"Binary predictions saved to {output_dir}/binary_predictions.csv")
    
    # Save metrics summary
    metrics_summary = {
        'technique': 'Zero-shot (Binary)',
        'primary_accuracy': metrics['accuracy'],
        'binary_accuracy': metrics['binary_accuracy'],
        'kappa': metrics['kappa'],
        'alpha': metrics['alpha'],
        'icc': metrics['icc']
    }
    
    summary_df = pd.DataFrame([metrics_summary])
    summary_df.to_csv(f"{output_dir}/metrics_summary.csv", index=False)
    
    return detailed_results, metrics

if __name__ == "__main__":
    # Import config
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config
    
    # Run evaluation
    try:
        print("\nStarting Zero-shot evaluation on Bloom taxonomy dataset (Binary Classification)...")
        print(f"Using data file: {config.DATA_PATH}")
        
        results, metrics = evaluate_zero_shot(
            data_path=config.DATA_PATH,
            api_key=config.OPENAI_API_KEY,
            limit=5  # Set to small number for testing
        )
        print("\nEvaluation completed successfully!")
        
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        import traceback
        print("\nFull error traceback:")
        print(traceback.format_exc())
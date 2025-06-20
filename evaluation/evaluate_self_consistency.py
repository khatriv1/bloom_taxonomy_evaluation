# bloom_taxonomy_evaluation/evaluation/evaluate_self_consistency.py

import sys
import os
import pandas as pd
import openai
import time
from typing import Dict, List

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompting.self_consistency import get_self_consistency_prediction_single_category
from utils.data_loader import load_and_preprocess_bloom_data, get_outcome_category, filter_valid_outcomes
from utils.metrics import calculate_agreement_metrics, plot_category_performance, print_detailed_results

def evaluate_self_consistency(data_path: str, api_key: str, output_dir: str = "results/self_consistency", limit: int = None):
    """Evaluate Self-Consistency prompting technique on Bloom taxonomy dataset."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess data
    try:
        df = load_and_preprocess_bloom_data(data_path)
        if df.empty:
            raise Exception("No valid examples found in the data file")
        
        # Filter to valid outcomes
        df = filter_valid_outcomes(df, min_score=0.1)
        
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
    human_labels = []  # Ground truth categories
    model_labels = []  # Model predictions
    detailed_results = []
    
    # Process each learning outcome
    total = len(df)
    for seq, (_, row) in enumerate(df.iterrows(), start=1):
        print(f"\nProcessing outcome {seq}/{total}")
        print(f"Learning outcome: {row['learning_outcome'][:100]}...")
        
        outcome_id = str(row['outcome_id'])
        
        try:
            # Get human ground truth (highest scoring category)
            human_category = get_outcome_category(row, method='highest_score')
            human_labels.append(human_category)
            
            # Get model predictions with Self-Consistency
            # Note: Using fewer samples (3) for faster evaluation
            model_category = get_self_consistency_prediction_single_category(
                learning_outcome=row['learning_outcome'],
                client=client,
                n_samples=3  # Reduced for efficiency
            )
            model_labels.append(model_category)
            
            # Store detailed result
            detailed_results.append({
                'outcome_id': outcome_id,
                'learning_outcome': row['learning_outcome'],
                'human_category': human_category,
                'model_category': model_category,
                'exact_match': human_category == model_category,
                'ground_truth_score': row['ground_truth_score']
            })
            
            print(f"Human: {human_category}")
            print(f"Model: {model_category}")
            print(f"Match: {human_category == model_category}")
            
        except Exception as e:
            print(f"Error processing outcome {outcome_id}: {str(e)}")
            continue
        
        time.sleep(1)  # Rate limiting
    
    if not human_labels:
        raise Exception("No valid predictions were generated")
    
    # Calculate metrics
    metrics = calculate_agreement_metrics(human_labels, model_labels, categories)
    
    # Create visualization
    plot_category_performance(
        metrics, 
        categories, 
        'Self-Consistency',
        f"{output_dir}/self_consistency_performance.png"
    )
    
    # Print results
    print_detailed_results(metrics, categories, 'Self-Consistency')
    
    # Save detailed results
    results_df = pd.DataFrame(detailed_results)
    results_df.to_csv(f"{output_dir}/detailed_results.csv", index=False)
    print(f"\nDetailed results saved to {output_dir}/detailed_results.csv")
    
    # Save metrics summary
    metrics_summary = {
        'technique': 'Self-Consistency',
        'accuracy': metrics['accuracy'],
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
        print("\nStarting Self-Consistency evaluation on Bloom taxonomy dataset...")
        print(f"Using data file: {config.DATA_PATH}")
        
        results, metrics = evaluate_self_consistency(
            data_path=config.DATA_PATH,
            api_key=config.OPENAI_API_KEY,
            limit=10  # Set to small number for testing
        )
        print("\nEvaluation completed successfully!")
        
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        import traceback
        print("\nFull error traceback:")
        print(traceback.format_exc())
# bloom_taxonomy_evaluation/evaluation/evaluate_active_prompt.py
# MINIMAL VERSION: Reduced parameters to match active_prompt.py

import sys
import os
import pandas as pd
import openai
import time
from typing import Dict, List

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompting.active_prompt import (
    get_active_prompt_prediction_binary_class, 
    get_active_prompt_prediction_single_category,
    prepare_active_prompting_data
)
from utils.data_loader import load_and_preprocess_bloom_data, get_outcome_category, get_outcome_binary_labels, filter_valid_outcomes
from utils.metrics import calculate_agreement_metrics, plot_category_performance, print_detailed_results

def evaluate_active_prompt(data_path: str, api_key: str, output_dir: str = "results/active_prompt", limit: int = None):
    """Evaluate Active Prompting technique on Bloom taxonomy dataset with MINIMAL parameters."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess data
    try:
        df = load_and_preprocess_bloom_data(data_path)
        if df.empty:
            raise Exception("No valid examples found in the data file")
        
        # Filter to valid outcomes
        df = filter_valid_outcomes(df, min_labels=1)
        
        print(f"Loaded {len(df)} total learning outcomes")
        
        if limit:
            # REDUCED uncertainty pool size
            uncertainty_size = min(20, max(10, len(df) // 4))  # REDUCED from min(1000, max(100, len(df) // 2))
            eval_size = min(limit, 15)  # Cap evaluation size
            
            # Sample for uncertainty estimation (smaller pool)
            uncertainty_df = df.sample(n=uncertainty_size, random_state=42)
            
            # Sample for evaluation
            remaining_df = df.drop(uncertainty_df.index) if len(df) > uncertainty_size + eval_size else df
            if len(remaining_df) >= eval_size:
                eval_df = remaining_df.sample(n=eval_size, random_state=43)
            else:
                eval_df = df.sample(n=eval_size, random_state=43)
                
            print(f"Using {len(uncertainty_df)} outcomes for uncertainty estimation (REDUCED)")
            print(f"Evaluating on {len(eval_df)} outcomes")
        else:
            # Default small sizes for testing
            uncertainty_size = min(20, len(df))
            eval_size = min(10, len(df))
            
            uncertainty_df = df.sample(n=uncertainty_size, random_state=42)
            eval_df = df.sample(n=eval_size, random_state=43)
            
            print(f"Using {len(uncertainty_df)} outcomes for uncertainty estimation")
            print(f"Evaluating on {len(eval_df)} outcomes")
        
    except Exception as e:
        raise Exception(f"Error loading or processing data: {str(e)}")
    
    # Initialize OpenAI client
    client = openai.OpenAI(api_key=api_key)
    
    # Bloom taxonomy categories
    categories = [
        'remember', 'understand', 'apply', 'analyze', 'evaluate', 'create'
    ]
    
    print("\n" + "="*60)
    print("ACTIVE PROMPTING: UNCERTAINTY ESTIMATION PHASE (MINIMAL)")
    print("="*60)
    print("Optimized parameters:")
    print(f"  • Pool size: {len(uncertainty_df)} outcomes")
    print(f"  • k_samples: 2 per outcome")
    print(f"  • Examples: 2 per category")
    print(f"  • Expected API calls: ~{len(uncertainty_df) * 2 * 6} for uncertainty")
    print("="*60)
    
    # STAGE 1-3: Uncertainty Estimation, Selection, and Annotation
    try:
        uncertainty_examples = prepare_active_prompting_data(uncertainty_df, client, n_examples=2)  # REDUCED from 8 to 2
        print(f"Active Prompting preparation completed with examples for {len(uncertainty_examples)} categories")
        
        # Show what was created
        for category, examples in uncertainty_examples.items():
            print(f"  {category}: {len(examples)} uncertain examples")
            
    except Exception as e:
        print(f"Uncertainty estimation failed: {e}")
        print("Using fallback method...")
        
        # Fallback: simple random selection
        sample_outcomes = uncertainty_df.sample(n=min(6, len(uncertainty_df)), random_state=42)
        uncertainty_examples = {}
        
        for category in categories:
            category_examples = []
            for _, row in sample_outcomes.head(2).iterrows():  # Only 2 examples
                binary_labels = get_outcome_binary_labels(row)
                label = binary_labels[category]
                reasoning = f"This {'involves' if label == 1 else 'does not involve'} {category} thinking. Answer: {label}"
                category_examples.append((row['learning_outcome'], reasoning))
            uncertainty_examples[category] = category_examples
        
        print(f"Fallback preparation completed with examples for all categories")
    
    print("\n" + "="*60)
    print("ACTIVE PROMPTING: EVALUATION PHASE (MINIMAL)")
    print("="*60)
    
    # Store results
    human_labels = []
    model_labels = []
    detailed_results = []
    all_model_binary = []
    all_human_binary = []
    
    # STAGE 4: Inference with Selected Examples
    total = len(eval_df)
    successful_predictions = 0
    
    for seq, (_, row) in enumerate(eval_df.iterrows(), start=1):
        print(f"\nProcessing outcome {seq}/{total}")
        print(f"Learning outcome: {row['learning_outcome'][:80]}...")
        
        outcome_id = str(row.get('outcome_id', seq))
        
        try:
            # Get human ground truth
            human_category = get_outcome_category(row, method='first_selected')
            human_labels.append(human_category)
            
            human_binary = get_outcome_binary_labels(row)
            all_human_binary.append(human_binary)
            
            # Get model predictions with Active Prompting
            model_binary = get_active_prompt_prediction_binary_class(
                learning_outcome=row['learning_outcome'],
                client=client,
                uncertainty_data=uncertainty_examples
            )
            all_model_binary.append(model_binary)
            
            # Get single best category
            model_category = get_active_prompt_prediction_single_category(
                learning_outcome=row['learning_outcome'],
                client=client,
                uncertainty_data=uncertainty_examples
            )
            model_labels.append(model_category)
            successful_predictions += 1
            
            # Calculate binary accuracy
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
            print(f"Binary accuracy: {binary_accuracy:.3f} ({binary_matches}/6)")
            
        except Exception as e:
            print(f"Error processing outcome {outcome_id}: {str(e)}")
            continue
        
        time.sleep(0.3)  # Faster rate limiting
    
    if not human_labels:
        raise Exception("No valid predictions were generated")
    
    print(f"\nSuccessfully processed {successful_predictions}/{total} outcomes")
    
    # Calculate metrics
    metrics = calculate_agreement_metrics(human_labels, model_labels, categories)
    
    # Calculate binary-level metrics
    total_binary_matches = sum(result['binary_matches'] for result in detailed_results)
    total_binary_comparisons = len(detailed_results) * len(categories)
    overall_binary_accuracy = total_binary_matches / total_binary_comparisons
    
    metrics['binary_accuracy'] = overall_binary_accuracy * 100
    print(f"\nOverall Binary Accuracy: {overall_binary_accuracy:.3f} ({total_binary_matches}/{total_binary_comparisons})")
    
    # Create visualization
    try:
        plot_category_performance(
            metrics, 
            categories, 
            'Active Prompting (MINIMAL)',
            f"{output_dir}/active_prompt_performance.png"
        )
        print(f"📊 Performance plot saved to {output_dir}/active_prompt_performance.png")
    except Exception as e:
        print(f"⚠️ Could not create plot: {e}")
    
    # Print results
    print_detailed_results(metrics, categories, 'Active Prompting (MINIMAL)')
    
    # Save detailed results
    results_df = pd.DataFrame(detailed_results)
    results_df.to_csv(f"{output_dir}/detailed_results.csv", index=False)
    print(f"💾 Detailed results saved to {output_dir}/detailed_results.csv")
    
    # Save metrics summary
    metrics_summary = {
        'technique': 'Active Prompting (MINIMAL)',
        'primary_accuracy': metrics['accuracy'],
        'binary_accuracy': metrics['binary_accuracy'],
        'kappa': metrics['kappa'],
        'alpha': metrics['alpha'],
        'icc': metrics['icc'],
        'uncertainty_pool_size': len(uncertainty_df),
        'successful_predictions': successful_predictions
    }
    
    summary_df = pd.DataFrame([metrics_summary])
    summary_df.to_csv(f"{output_dir}/metrics_summary.csv", index=False)
    print(f"📊 Metrics summary saved to {output_dir}/metrics_summary.csv")
    
    return detailed_results, metrics

if __name__ == "__main__":
    # Import config
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config
    
    # Run evaluation
    try:
        print("Starting MINIMAL Active Prompting evaluation on Bloom taxonomy dataset...")
        print("Optimized for fast testing:")
        print("   • Small uncertainty pool (20 outcomes)")
        print("   • Fewer samples (k=2 vs k=10)")
        print("   • Fewer examples (2 vs 8 per category)")
        print("   • Simplified prompts")
        print(f"Using data file: {config.DATA_PATH}")
        
        # Ask user for limit
        try:
            limit_input = input("\nEnter number of outcomes to evaluate (or press Enter for 10): ").strip()
            limit = int(limit_input) if limit_input else 10
            
            if limit > 15:
                print(f"Reducing limit from {limit} to 15 for faster testing")
                limit = 15
                
        except ValueError:
            limit = 10
        
        print(f"\n🎯 Configuration:")
        print(f"   • Uncertainty pool: ~20 outcomes")
        print(f"   • Evaluation set: {limit} outcomes")
        print(f"   • Expected API calls: ~240 for uncertainty + ~{limit * 6} for evaluation")
        print(f"   • Estimated time: 2-3 minutes")
        
        # Confirm before starting
        confirm = input("\n▶Continue? (y/n): ").strip().lower()
        if confirm not in ['y', 'yes', '']:
            print("Cancelled")
            exit()
        
        start_time = time.time()
        
        results, metrics = evaluate_active_prompt(
            data_path=config.DATA_PATH,
            api_key=config.OPENAI_API_KEY,
            limit=limit
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print("\n" + "="*60)
        print("MINIMAL ACTIVE PROMPTING EVALUATION COMPLETED!")
        print("="*60)
        print(f"Total time: {duration:.1f} seconds")
        print(f"Primary Accuracy: {metrics['accuracy']:.1%}")
        print(f"Binary Accuracy: {metrics['binary_accuracy']:.1%}")
        print(f"Cohen's Kappa: {metrics['kappa']:.3f}")
        print(f"Results saved in: results/active_prompt/")
        
    except KeyboardInterrupt:
        print("\n\nEvaluation stopped by user")
        
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        print("\nTroubleshooting tips:")
        print("   • Check your OpenAI API key")
        print("   • Ensure data file exists")
        print("   • Try with smaller limit (e.g., 5)")
        
        import traceback
        print("\n📋 Full error traceback:")
        print(traceback.format_exc())
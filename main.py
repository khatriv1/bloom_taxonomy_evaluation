# bloom_taxonomy_evaluation/main.py

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Import all evaluation functions
from evaluation.evaluate_zero_shot import evaluate_zero_shot
from evaluation.evaluate_cot import evaluate_cot
from evaluation.evaluate_few_shot import evaluate_few_shot
from evaluation.evaluate_active_prompt import evaluate_active_prompt
from evaluation.evaluate_auto_cot import evaluate_auto_cot
from evaluation.evaluate_contrastive_cot import evaluate_contrastive_cot
from evaluation.evaluate_rephrase_respond import evaluate_rephrase_respond
from evaluation.evaluate_self_consistency import evaluate_self_consistency
from evaluation.evaluate_take_step_back import evaluate_take_step_back

import config

def create_comparison_visualization(comparison_df, output_dir):
    """Create comparison visualizations for all Bloom techniques using the 4 metrics."""
    plt.style.use('default')
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14))
    
    # Prepare data
    techniques = comparison_df['Technique'].tolist()
    x = np.arange(len(techniques))
    
    # First subplot: All 4 metrics comparison
    metrics = ['Accuracy (%)', 'Cohen\'s κ', 'Krippendorff\'s α', 'ICC']
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    width = 0.2
    
    for i, metric in enumerate(metrics):
        offset = width * i
        if metric == 'Accuracy (%)':
            values = comparison_df['Accuracy']
        elif metric == 'Cohen\'s κ':
            values = comparison_df['Kappa'] * 100  # Scale for visibility
        elif metric == 'Krippendorff\'s α':
            values = comparison_df['Alpha'] * 100
        else:  # ICC
            values = comparison_df['ICC'] * 100
        
        bars = ax1.bar(x + offset, values, width, label=metric, color=colors[i], alpha=0.8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if metric == 'Accuracy (%)':
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=8)
            else:
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height/100:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax1.set_xlabel('Prompting Technique', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Bloom Taxonomy Classification: 4 Metrics Comparison',
                 fontsize=14, fontweight='bold', pad=20)
    ax1.set_ylim(0, 110)
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(techniques, rotation=45, ha='right')
    ax1.legend(loc='upper right')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # Second subplot: Ranking by Cohen's Kappa
    sorted_df = comparison_df.sort_values('Kappa', ascending=False)
    y_pos = np.arange(len(sorted_df))
    
    bars = ax2.barh(y_pos, sorted_df['Kappa'], alpha=0.8, 
                    color=['#2ecc71' if i == 0 else '#3498db' for i in range(len(sorted_df))])
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(sorted_df['Technique'])
    ax2.set_xlabel('Cohen\'s Kappa (κ)', fontsize=12, fontweight='bold')
    ax2.set_title('Techniques Ranked by Agreement (Cohen\'s κ)', fontsize=14, fontweight='bold', pad=20)
    ax2.grid(True, axis='x', linestyle='--', alpha=0.3)
    ax2.set_xlim(0, 1)
    
    # Add value labels
    for i, (idx, row) in enumerate(sorted_df.iterrows()):
        kappa = row['Kappa']
        ax2.text(kappa + 0.01, i, f'{kappa:.3f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/bloom_all_techniques_comparison.png", 
                dpi=300, bbox_inches='tight')
    plt.close()

def run_bloom_evaluations(data_path, api_key, output_dir="results", limit=None, techniques=None):
    """Run Bloom taxonomy classification evaluations for all techniques."""
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"bloom_evaluation_all_techniques_{timestamp}"
    output_path = os.path.join(output_dir, base_dir)
    
    # Handle existing directories
    counter = 1
    while os.path.exists(output_path):
        output_path = os.path.join(output_dir, f"{base_dir}_{counter}")
        counter += 1
    
    os.makedirs(output_path, exist_ok=True)
    print(f"\nBloom taxonomy evaluation results will be saved in: {output_path}")
    
    # Store results from each technique
    all_results = {}
    detailed_results = []
    
    # All available techniques with their evaluation functions
    if techniques is None:
        techniques = {
            'Zero-shot': evaluate_zero_shot,
            'Chain of Thought': evaluate_cot,
            'Few-shot': evaluate_few_shot,
            'Active Prompting': evaluate_active_prompt,
            'Auto-CoT': evaluate_auto_cot,
            'Contrastive CoT': evaluate_contrastive_cot,
            'Rephrase and Respond': evaluate_rephrase_respond,
            'Self-Consistency': evaluate_self_consistency,
            'Take a Step Back': evaluate_take_step_back
        }
    
    # Process each technique
    for technique_name, evaluate_func in techniques.items():
        print(f"\n{'='*60}")
        print(f"Running {technique_name} evaluation...")
        print(f"{'='*60}")
        
        # Create technique-specific directory
        technique_dir = os.path.join(output_path, technique_name.lower().replace(' ', '_'))
        os.makedirs(technique_dir, exist_ok=True)
        
        try:
            # Run evaluation
            results, metrics = evaluate_func(data_path, api_key,
                                          output_dir=technique_dir, limit=limit)
            all_results[technique_name] = metrics
            
            # Collect detailed results
            results_df = pd.DataFrame(results)
            results_df['Technique'] = technique_name
            detailed_results.append(results_df)
            
            print(f"✓ {technique_name} completed successfully")
            
        except Exception as e:
            print(f"✗ {technique_name} failed: {str(e)}")
            continue
    
    if not all_results:
        print("No successful evaluations completed!")
        return None, None
    
    # Create comparison DataFrame with the 4 metrics
    comparison_data = []
    for technique_name, metrics in all_results.items():
        comparison_data.append({
            'Technique': technique_name,
            'Accuracy': metrics.get('accuracy', 0),
            'Kappa': metrics.get('kappa', 0),
            'Alpha': metrics.get('alpha', 0),
            'ICC': metrics.get('icc', 0)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save comparison results
    comparison_df.to_csv(f"{output_path}/bloom_all_techniques_comparison.csv", index=False)
    
    # Create comparison visualization
    create_comparison_visualization(comparison_df, output_path)
    
    # Combine detailed results
    if detailed_results:
        all_detailed_results = pd.concat(detailed_results, ignore_index=True)
        all_detailed_results.to_csv(f"{output_path}/bloom_all_detailed_results.csv", index=False)
    
    # Generate comprehensive summary report
    with open(f"{output_path}/bloom_comprehensive_report.txt", 'w') as f:
        f.write("=== Bloom Taxonomy Classification: Comprehensive Evaluation Report ===\n\n")
        
        f.write("Dataset: Learning Outcomes with Bloom Taxonomy Classifications\n")
        f.write("Task: Single-label classification of learning outcomes into 6 Bloom categories\n")
        f.write(f"Number of outcomes evaluated: {limit if limit else 'All available'}\n")
        f.write(f"Total techniques evaluated: {len(all_results)}\n\n")
        
        # Overall metrics comparison
        f.write("=== Overall Metrics Comparison ===\n")
        f.write("Metrics Used:\n")
        f.write("- Accuracy: How often the model exactly matches human expert classifications\n")
        f.write("- Cohen's Kappa (κ): Agreement beyond chance (-1 to 1, higher is better)\n")
        f.write("- Krippendorff's Alpha (α): Reliability measure (0 to 1, higher is better)\n")
        f.write("- ICC: Intraclass Correlation Coefficient (-1 to 1, higher is better)\n\n")
        
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")
        
        # Best performing technique for each metric
        f.write("=== Best Performing Techniques by Metric ===\n")
        for metric in ['Accuracy', 'Kappa', 'Alpha', 'ICC']:
            best_technique = comparison_df.loc[comparison_df[metric].idxmax()]
            metric_name = {
                'Accuracy': 'Accuracy',
                'Kappa': 'Cohen\'s Kappa (κ)',
                'Alpha': 'Krippendorff\'s Alpha (α)',
                'ICC': 'Intraclass Correlation (ICC)'
            }[metric]
            if metric == 'Accuracy':
                f.write(f"{metric_name}: {best_technique['Technique']} ({best_technique[metric]:.1f}%)\n")
            else:
                f.write(f"{metric_name}: {best_technique['Technique']} ({best_technique[metric]:.3f})\n")
        
        # Bloom Categories explanation
        f.write("\n\n=== Bloom Taxonomy Category Definitions ===\n\n")
        category_descriptions = {
            "remember": "Retrieving relevant knowledge from long-term memory",
            "understand": "Determining the meaning of instructional messages", 
            "apply": "Carrying out or using a procedure in a given situation",
            "analyze": "Breaking material into constituent parts and detecting relationships",
            "evaluate": "Making judgments based on criteria and standards",
            "create": "Putting elements together to form a novel, coherent whole"
        }
        
        for category, description in category_descriptions.items():
            f.write(f"{category}: {description}\n")
    
    print(f"\n{'='*70}")
    print("BLOOM TAXONOMY COMPREHENSIVE EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved in: {output_path}")
    print("\nGenerated files:")
    print("- bloom_all_techniques_comparison.csv (Overall metrics for all techniques)")
    print("- bloom_all_techniques_comparison.png (Visual comparison)")
    print("- bloom_all_detailed_results.csv (All detailed predictions)")
    print("- bloom_comprehensive_report.txt (Complete analysis report)")
    print("- Individual technique results in subdirectories")
    
    return comparison_df, detailed_results

if __name__ == "__main__":
    print("=" * 70)
    print("Bloom Taxonomy Classification: Comprehensive Evaluation Suite")
    print("=" * 70)
    print(f"Using Bloom taxonomy dataset: {config.DATA_PATH}")
    
    # Ask user for evaluation parameters
    print("\nEvaluation Options:")
    print("This will evaluate different prompting strategies on the Bloom taxonomy dataset.")
    print("\nMetrics to be calculated:")
    print("- Accuracy: Exact match between model and human expert classifications")
    print("- Cohen's Kappa (κ): Agreement beyond chance")
    print("- Krippendorff's Alpha (α): Reliability measure")
    print("- Intraclass Correlation (ICC): Score correlation")
    
    limit_input = input("\nEnter number of learning outcomes to evaluate (recommended: 50-100 for testing, or press Enter for all): ")
    
    if limit_input.strip():
        try:
            limit = int(limit_input)
            if limit < 10:
                print("WARNING: Very small sample size may not provide reliable results.")
        except ValueError:
            print("Invalid input. Using all available learning outcomes.")
            limit = None
    else:
        limit = None
    
    # Ask which techniques to evaluate
    print("\nAvailable prompting techniques:")
    all_techniques = [
        "1. Zero-shot",
        "2. Chain of Thought", 
        "3. Few-shot",
        "4. Active Prompting",
        "5. Auto-CoT",
        "6. Contrastive CoT",
        "7. Rephrase and Respond",
        "8. Self-Consistency",
        "9. Take a Step Back",
        "10. All techniques"
    ]
    
    for technique in all_techniques:
        print(technique)
    
    technique_input = input("\nEnter technique numbers (comma-separated) or 10 for all: ")
    
    try:
        if not technique_input.strip() or '10' in technique_input:
            selected_techniques = None
        else:
            selected_indices = [int(idx.strip()) for idx in technique_input.split(",")]
            technique_map = {
                1: ("Zero-shot", evaluate_zero_shot),
                2: ("Chain of Thought", evaluate_cot),
                3: ("Few-shot", evaluate_few_shot),
                4: ("Active Prompting", evaluate_active_prompt),
                5: ("Auto-CoT", evaluate_auto_cot),
                6: ("Contrastive CoT", evaluate_contrastive_cot),
                7: ("Rephrase and Respond", evaluate_rephrase_respond),
                8: ("Self-Consistency", evaluate_self_consistency),
                9: ("Take a Step Back", evaluate_take_step_back),
            }
            
            selected_techniques = {}
            for idx in selected_indices:
                if idx in technique_map:
                    name, func = technique_map[idx]
                    selected_techniques[name] = func
    
    except ValueError:
        print("Invalid input. Running all techniques.")
        selected_techniques = None
    
    # Run evaluations
    try:
        print("\nStarting comprehensive Bloom taxonomy evaluation...")
        print("This may take some time depending on the number of outcomes and techniques selected.")
        
        comparison_df, detailed_results = run_bloom_evaluations(
            data_path=config.DATA_PATH,
            api_key=config.OPENAI_API_KEY,
            limit=limit,
            techniques=selected_techniques
        )
        
        if comparison_df is not None:
            print("\nAll evaluations completed successfully!")
            print("\nTop techniques by Cohen's Kappa:")
            top_techniques = comparison_df.nlargest(3, 'Kappa')
            for rank, (idx, row) in enumerate(top_techniques.iterrows(), 1):
                print(f"{rank}. {row['Technique']}: κ={row['Kappa']:.3f}")
        else:
            print("\nNo evaluations completed successfully.")
            
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        import traceback
        print("\nFull error traceback:")
        print(traceback.format_exc())
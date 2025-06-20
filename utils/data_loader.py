# bloom_taxonomy_evaluation/utils/data_loader.py

"""
Data loading utilities for the Bloom Taxonomy dataset.
Handles learning outcomes with Bloom taxonomy classifications.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional

def load_and_preprocess_bloom_data(file_path: str):
    """
    Load and preprocess Bloom taxonomy data.
    """
    print(f"Loading data from: {file_path}")
    
    # Read CSV file
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    except Exception as e:
        raise Exception(f"Error reading CSV file: {str(e)}")
    
    # Process Bloom taxonomy format
    print("Processing Bloom taxonomy dataset format.")
    return process_bloom_format(df)

def process_bloom_format(df):
    """Process Bloom taxonomy dataset format."""
    processed_data = []
    
    # The 6 Bloom taxonomy categories
    categories = {
        'remember': 'Remember',
        'understand': 'Understand', 
        'apply': 'Apply',
        'analyze': 'Analyze',
        'evaluate': 'Evaluate',
        'create': 'Create'
    }
    
    # Process each learning outcome
    for idx, row in df.iterrows():
        if pd.isna(row.get('Learning_outcome', None)):
            continue
            
        # Basic learning outcome info
        outcome_data = {
            'outcome_id': idx,
            'learning_outcome': row['Learning_outcome'],
            'datasplit': 'train'  # Default split
        }
        
        # Extract expert scores for each Bloom category
        for category_key, category_name in categories.items():
            score = row.get(category_name, 0.0)
            outcome_data[f'expert_{category_key}'] = float(score) if not pd.isna(score) else 0.0
        
        # Determine ground truth category (highest scoring category)
        scores = [outcome_data[f'expert_{cat}'] for cat in categories.keys()]
        max_idx = np.argmax(scores)
        ground_truth_category = list(categories.keys())[max_idx]
        outcome_data['ground_truth_category'] = ground_truth_category
        outcome_data['ground_truth_score'] = max(scores)
        
        processed_data.append(outcome_data)
    
    result_df = pd.DataFrame(processed_data)
    print(f"\nProcessed {len(result_df)} learning outcomes with expert scores")
    
    # Print distribution statistics
    print("\nGround truth category distribution:")
    for category in categories.keys():
        count = (result_df['ground_truth_category'] == category).sum()
        total = len(result_df)
        print(f"  {category}: {count}/{total} ({count/total*100:.1f}%)")
    
    return result_df

def get_outcome_category(row: pd.Series, method: str = 'highest_score') -> str:
    """
    Extract the primary category for a learning outcome.
    
    Args:
        row: A row from the dataframe
        method: Method to determine category ('highest_score', 'threshold')
    
    Returns:
        Primary Bloom category name
    """
    categories = ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']
    
    if method == 'highest_score':
        # Return the category with the highest expert score
        return row['ground_truth_category']
    
    elif method == 'threshold':
        # Return categories above a threshold (for multi-label)
        assigned_categories = []
        for category in categories:
            score_col = f'expert_{category}'
            if score_col in row and row[score_col] >= 0.5:  # Threshold of 0.5
                assigned_categories.append(category)
        return assigned_categories if assigned_categories else ['remember']  # Default fallback
    
    return row['ground_truth_category']

def filter_valid_outcomes(df: pd.DataFrame, min_score: float = 0.1) -> pd.DataFrame:
    """
    Filter to keep only outcomes with meaningful expert scores.
    """
    categories = ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']
    
    # Check that each outcome has at least one category with a reasonable score
    valid_outcomes = []
    for idx, row in df.iterrows():
        max_score = max([row[f'expert_{cat}'] for cat in categories])
        if max_score >= min_score:
            valid_outcomes.append(idx)
    
    filtered_df = df.loc[valid_outcomes].copy()
    
    print(f"Filtered to {len(filtered_df)} outcomes with scores >= {min_score}")
    return filtered_df

def get_bloom_statistics(df: pd.DataFrame) -> Dict:
    """
    Get statistics about the Bloom taxonomy dataset.
    """
    categories = ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']
    
    stats = {
        'total_outcomes': len(df),
        'category_distribution': {},
        'average_scores': {},
        'score_ranges': {}
    }
    
    # Category distribution
    for category in categories:
        count = (df['ground_truth_category'] == category).sum()
        stats['category_distribution'][category] = {
            'count': count,
            'percentage': count / len(df) * 100
        }
    
    # Average expert scores
    for category in categories:
        score_col = f'expert_{category}'
        if score_col in df.columns:
            scores = df[score_col]
            stats['average_scores'][category] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'min': scores.min(),
                'max': scores.max()
            }
    
    return stats
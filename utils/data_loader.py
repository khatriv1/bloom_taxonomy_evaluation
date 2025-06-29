# bloom_taxonomy_evaluation/utils/data_loader.py

"""
Data loading utilities for the Bloom Taxonomy dataset.
Handles learning outcomes with BINARY Bloom taxonomy classifications (0 and 1 only).
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
    print("Processing Bloom taxonomy dataset format (BINARY data: 0s and 1s only).")
    return process_bloom_format(df)

def process_bloom_format(df):
    """Process Bloom taxonomy dataset format for BINARY data (0s and 1s)."""
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
    
    print(f"Processing {len(df)} learning outcomes...")
    
    # Process each learning outcome
    for idx, row in df.iterrows():
        if pd.isna(row.get('Learning_outcome', None)) or str(row.get('Learning_outcome', '')).strip() == '':
            continue
            
        # Basic learning outcome info
        outcome_data = {
            'outcome_id': idx,
            'learning_outcome': str(row['Learning_outcome']).strip(),
            'datasplit': 'train'  # Default split
        }
        
        # Extract BINARY values for each Bloom category (handle NaN/empty as 0)
        for category_key, category_name in categories.items():
            value = row.get(category_name, 0)
            
            # Handle NaN, None, empty values -> convert to 0
            if pd.isna(value) or value is None:
                binary_value = 0
            else:
                # Convert to int and ensure it's 0 or 1
                try:
                    binary_value = int(float(value))  # Handle both int and float input
                    if binary_value not in [0, 1]:
                        print(f"Warning: Non-binary value {value} for {category_name} in row {idx}, setting to 0")
                        binary_value = 0
                except (ValueError, TypeError):
                    print(f"Warning: Invalid value '{value}' for {category_name} in row {idx}, setting to 0")
                    binary_value = 0
            
            outcome_data[f'expert_{category_key}'] = binary_value
        
        # Determine ground truth category (first category with value 1)
        binary_values = [outcome_data[f'expert_{cat}'] for cat in categories.keys()]
        
        if sum(binary_values) == 0:
            # No category selected - use 'understand' as default
            ground_truth_category = 'understand'
        elif sum(binary_values) == 1:
            # Exactly one category - use it
            ground_truth_category = list(categories.keys())[binary_values.index(1)]
        else:
            # Multiple categories - use first one with value 1
            ground_truth_category = list(categories.keys())[binary_values.index(1)]
        
        outcome_data['ground_truth_category'] = ground_truth_category
        outcome_data['ground_truth_score'] = 1  # Always 1 for binary data
        
        processed_data.append(outcome_data)
    
    result_df = pd.DataFrame(processed_data)
    print(f"\nProcessed {len(result_df)} valid learning outcomes with binary labels")
    
    # Check for data quality issues
    print("\nData Quality Check:")
    for category in categories.keys():
        score_col = f'expert_{category}'
        values = result_df[score_col]
        ones_count = (values == 1).sum()
        zeros_count = (values == 0).sum()
        print(f"  {category:12s}: {ones_count:5d} ones, {zeros_count:5d} zeros ({ones_count/(ones_count+zeros_count)*100:.1f}% positive)")
    
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
        method: Method to determine category ('highest_score', 'first_selected')
    
    Returns:
        Primary Bloom category name
    """
    categories = ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']
    
    if method == 'highest_score' or method == 'first_selected':
        # Return the precomputed ground truth category
        return row['ground_truth_category']
    
    elif method == 'threshold':
        # For binary data, return categories with value 1
        assigned_categories = []
        for category in categories:
            score_col = f'expert_{category}'
            if score_col in row and row[score_col] == 1:
                assigned_categories.append(category)
        return assigned_categories if assigned_categories else ['understand']  # Default fallback
    
    return row['ground_truth_category']

def get_outcome_binary_labels(row: pd.Series) -> Dict[str, int]:
    """
    Extract binary labels for all Bloom categories from a learning outcome row.
    NO THRESHOLD NEEDED - data is already binary (0 and 1)!
    
    Args:
        row: A row from the dataframe containing binary expert labels
    
    Returns:
        Dictionary with 1/0 labels for each category
    """
    categories = ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']
    binary_labels = {}
    
    for category in categories:
        score_col = f'expert_{category}'
        if score_col in row:
            # Data is already binary - just ensure it's int
            value = row[score_col]
            if pd.isna(value):
                binary_labels[category] = 0
            else:
                binary_labels[category] = int(value)
        else:
            # If expert score not available, default to 0
            binary_labels[category] = 0
    
    return binary_labels

def filter_valid_outcomes(df: pd.DataFrame, min_labels: int = 1) -> pd.DataFrame:
    """
    Filter to keep only outcomes with meaningful binary labels.
    
    Args:
        df: DataFrame with learning outcomes
        min_labels: Minimum number of categories that should be labeled as 1
    
    Returns:
        Filtered DataFrame
    """
    categories = ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']
    
    # Check that each outcome has at least min_labels categories with value 1
    valid_outcomes = []
    for idx, row in df.iterrows():
        # Check number of positive labels
        binary_labels = get_outcome_binary_labels(row)
        num_positive = sum(binary_labels.values())
        if num_positive >= min_labels:
            valid_outcomes.append(idx)
    
    filtered_df = df.loc[valid_outcomes].copy()
    
    print(f"Filtered to {len(filtered_df)} outcomes with >= {min_labels} positive labels")
    return filtered_df

def get_bloom_statistics(df: pd.DataFrame) -> Dict:
    """
    Get statistics about the Bloom taxonomy dataset.
    """
    categories = ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']
    
    stats = {
        'total_outcomes': len(df),
        'category_distribution': {},
        'binary_distribution': {}
    }
    
    # Category distribution (primary categories)
    for category in categories:
        count = (df['ground_truth_category'] == category).sum()
        stats['category_distribution'][category] = {
            'count': count,
            'percentage': count / len(df) * 100
        }
    
    # Binary label distribution
    all_binary_labels = []
    for _, row in df.iterrows():
        binary_labels = get_outcome_binary_labels(row)
        all_binary_labels.append(binary_labels)
    
    for category in categories:
        positive_count = sum(1 for labels in all_binary_labels if labels[category] == 1)
        stats['binary_distribution'][category] = {
            'positive_count': positive_count,
            'positive_percentage': positive_count / len(df) * 100,
            'negative_count': len(df) - positive_count,
            'negative_percentage': (len(df) - positive_count) / len(df) * 100
        }
    
    return stats

def check_data_format(df: pd.DataFrame):
    """
    Check if data is actually binary (0s and 1s only).
    """
    print("=== BINARY DATA FORMAT CHECK ===")
    
    categories = ['Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', 'Create']
    
    print(f"Total rows: {len(df)}")
    print(f"Learning outcomes with text: {df['Learning_outcome'].notna().sum()}")
    print()
    
    all_binary = True
    for cat in categories:
        if cat in df.columns:
            col_data = df[cat].dropna()  # Ignore NaN for this check
            unique_values = sorted(col_data.unique())
            
            print(f"{cat:12s}: unique values = {unique_values}")
            
            # Check if all values are 0 or 1
            if not all(val in [0, 1, 0.0, 1.0] for val in unique_values):
                all_binary = False
                print(f"{'':14s}  ⚠️  Contains non-binary values!")
            else:
                ones = (col_data == 1).sum()
                zeros = (col_data == 0).sum()
                print(f"{'':14s}  ✅ Binary: {ones} ones, {zeros} zeros")
        else:
            print(f"{cat:12s}: MISSING COLUMN")
            all_binary = False
    
    print(f"\n=== RESULT ===")
    if all_binary:
        print("✅ Data is BINARY (0s and 1s only) - NO threshold needed")
    else:
        print("❌ Data contains non-binary values - threshold needed")
    
    return all_binary

if __name__ == "__main__":
    # Quick test
    import sys
    if len(sys.argv) > 1:
        df = pd.read_csv(sys.argv[1])
        check_data_format(df)
    else:
        print("Usage: python data_loader.py path/to/data.csv")
# bloom_taxonomy_evaluation/prompting/self_consistency.py

"""
Self-Consistency prompting for Bloom taxonomy classification.
Binary version: Samples multiple reasoning paths and takes majority vote for each category.
"""

import time
import json
import re
import numpy as np
from typing import List, Optional, Dict
from collections import Counter
from utils.bloom_rubric import BloomRubric

def get_single_reasoning_path_binary(learning_outcome: str,
                                   client,
                                   temperature: float = 0.7) -> Optional[Dict[str, int]]:
    """
    Get a single reasoning path for binary classification.
    
    Args:
        learning_outcome: The learning outcome text
        client: OpenAI client
        temperature: Sampling temperature for diversity
    
    Returns:
        Dictionary with binary decisions for all categories or None if failed
    """
    categories = ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']
    
    rubric = BloomRubric()
    category_definitions = rubric.get_category_definitions()
    
    definitions_text = ""
    for cat in categories:
        definitions_text += f"{cat.upper()}: {category_definitions[cat]['description']}\n"

    prompt = f"""You are an expert educator classifying learning outcomes according to Bloom's Taxonomy.

BLOOM'S TAXONOMY CATEGORIES:
{definitions_text}

Learning Outcome: "{learning_outcome}"

Think through this step-by-step and provide binary decisions for each category:
1. What is the main action verb or cognitive demand in this outcome?
2. What cognitive processes does this require from students?
3. How does this match the Bloom taxonomy categories?
4. Which categories apply (1) and which don't (0)?

Provide your classification as binary decisions in this JSON format:

{{
    "remember": 0,
    "understand": 0,
    "apply": 0,
    "analyze": 0,
    "evaluate": 0,
    "create": 0
}}

Classification:"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert at classifying learning outcomes using Bloom's Taxonomy. Provide reasoning and JSON binary decisions."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=300
        )
        
        result = response.choices[0].message.content.strip()
        
        # Extract JSON from response
        binary_decisions = parse_binary_decisions(result, categories)
        return binary_decisions
            
    except Exception as e:
        print(f"Error in reasoning path: {str(e)}")
        return None

def get_self_consistency_prediction_binary_class(learning_outcome: str,
                                               client,
                                               n_samples: int = 5) -> Dict[str, int]:
    """
    Get Self-Consistency prediction using multiple reasoning paths.
    
    Args:
        learning_outcome: The learning outcome text
        client: OpenAI client
        n_samples: Number of reasoning paths to sample
    
    Returns:
        Dictionary with binary decisions based on majority vote
    """
    categories = ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']
    
    # Collect predictions from multiple reasoning paths
    all_decisions = []
    
    for i in range(n_samples):
        # Vary temperature for diversity
        temp = 0.5 + (i * 0.1)  # 0.5, 0.6, 0.7, 0.8, 0.9
        
        decisions = get_single_reasoning_path_binary(learning_outcome, client, temp)
        
        if decisions is not None:
            all_decisions.append(decisions)
        
        # Small delay between samples
        time.sleep(0.2)
    
    if not all_decisions:
        print(f"No valid predictions obtained for self-consistency")
        return {cat: 0 for cat in categories}
    
    # Take majority vote for each category
    majority_decisions = {}
    for cat in categories:
        cat_votes = [decisions[cat] for decisions in all_decisions]
        majority_vote = 1 if sum(cat_votes) > len(cat_votes) / 2 else 0
        majority_decisions[cat] = majority_vote
    
    print(f"Self-consistency: Used {len(all_decisions)} predictions")
    
    return majority_decisions

def get_self_consistency_prediction_single_category(learning_outcome: str,
                                                   client,
                                                   n_samples: int = 5) -> str:
    """Get single best category using self-consistency binary decisions."""
    binary_decisions = get_self_consistency_prediction_binary_class(learning_outcome, client, n_samples)
    
    selected_categories = [cat for cat, decision in binary_decisions.items() if decision == 1]
    
    if selected_categories:
        return selected_categories[0]
    else:
        return 'understand'

def parse_binary_decisions(response_text: str, categories: list) -> Dict[str, int]:
    """Parse binary decisions from model response."""
    decisions = {}
    
    try:
        json_match = re.search(r'\{[^}]+\}', response_text)
        if json_match:
            json_str = json_match.group()
            parsed = json.loads(json_str)
            
            for cat in categories:
                value = parsed.get(cat, 0)
                decisions[cat] = 1 if value == 1 else 0
        else:
            for cat in categories:
                pattern = rf'"{cat}"\s*:\s*([01])'
                match = re.search(pattern, response_text)
                if match:
                    decisions[cat] = int(match.group(1))
                else:
                    decisions[cat] = 0
                    
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing binary decisions: {e}")
        decisions = {cat: 0 for cat in categories}
    
    return decisions

# Alias for backward compatibility
def get_self_consistency_prediction_multi_class(learning_outcome: str, client, n_samples: int = 5) -> Dict[str, int]:
    """Alias for binary classification to maintain compatibility."""
    return get_self_consistency_prediction_binary_class(learning_outcome, client, n_samples)

# Legacy function for backwards compatibility
def get_single_reasoning_path(learning_outcome: str,
                            client,
                            temperature: float = 0.7) -> Optional[str]:
    """
    Legacy function - now uses binary approach.
    """
    binary_decisions = get_single_reasoning_path_binary(learning_outcome, client, temperature)
    if binary_decisions:
        selected_categories = [cat for cat, decision in binary_decisions.items() if decision == 1]
        return selected_categories[0] if selected_categories else 'understand'
    else:
        return None
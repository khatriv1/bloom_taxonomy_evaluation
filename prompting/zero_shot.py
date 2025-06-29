# bloom_taxonomy_evaluation/prompting/zero_shot.py

"""
Zero-shot prompting for Bloom taxonomy classification.
Binary version: Returns 1/0 decisions for each category (like human experts).
"""

import time
import json
import re
from typing import Optional, Dict
from utils.bloom_rubric import BloomRubric

def parse_binary_decisions(response_text: str, categories: list) -> Dict[str, int]:
    """
    Parse binary decisions from model response.
    
    Args:
        response_text: Raw response from the model
        categories: List of category names
        
    Returns:
        Dictionary with 1/0 decisions for each category
    """
    decisions = {}
    
    try:
        # Try to parse as JSON first
        json_match = re.search(r'\{[^}]+\}', response_text)
        if json_match:
            json_str = json_match.group()
            parsed = json.loads(json_str)
            
            for cat in categories:
                value = parsed.get(cat, 0)
                # Ensure it's 1 or 0
                decisions[cat] = 1 if value == 1 else 0
        else:
            # Fallback: parse line by line
            for cat in categories:
                pattern = rf'"{cat}"\s*:\s*([01])'
                match = re.search(pattern, response_text)
                if match:
                    decisions[cat] = int(match.group(1))
                else:
                    decisions[cat] = 0
                    
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing binary decisions: {e}")
        # Default fallback - all 0s
        decisions = {cat: 0 for cat in categories}
    
    return decisions

def get_zero_shot_prediction_binary_class(learning_outcome: str, client) -> Dict[str, int]:
    """
    Get zero-shot binary prediction for all 6 Bloom categories.
    
    Args:
        learning_outcome: The learning outcome text
        client: OpenAI client
    
    Returns:
        Dictionary with 1/0 decisions for all 6 categories (like human experts)
    """
    categories = ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']
    
    rubric = BloomRubric()
    category_definitions = rubric.get_category_definitions()
    
    # Create comprehensive prompt for binary classification
    definitions_text = ""
    for cat in categories:
        definitions_text += f"\n{cat.upper()}: {category_definitions[cat]['description']}\n"
        definitions_text += f"Examples: {', '.join(category_definitions[cat]['examples'][:2])}\n"
    
    prompt = f"""You are an expert educator classifying learning outcomes according to Bloom's Taxonomy.

BLOOM'S TAXONOMY CATEGORIES:
{definitions_text}

TASK: Analyze the following learning outcome and decide which Bloom taxonomy categories apply. For each category, respond with 1 if it applies or 0 if it doesn't apply.

Learning Outcome: "{learning_outcome}"

INSTRUCTIONS:
1. Analyze the learning outcome carefully
2. For each of the 6 Bloom taxonomy categories, decide: Does this learning outcome require this type of thinking?
3. Respond with 1 (yes, this category applies) or 0 (no, this category doesn't apply)
4. Multiple categories can be 1 if the learning outcome involves multiple types of thinking
5. Respond in this exact JSON format:

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
                {"role": "system", "content": "You are an expert at classifying learning outcomes using Bloom's Taxonomy. Respond only with valid JSON containing 1 or 0 for each category."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=150
        )
        
        result = response.choices[0].message.content.strip()
        
        # Extract JSON from response
        binary_decisions = parse_binary_decisions(result, categories)
        
        return binary_decisions
        
    except Exception as e:
        print(f"Error in zero-shot binary classification: {str(e)}")
        # Return default - no categories selected
        return {cat: 0 for cat in categories}

def get_zero_shot_prediction_single_category(learning_outcome: str, client) -> str:
    """
    Get zero-shot prediction for the single best Bloom category.
    Uses binary decisions and returns the first category marked as 1, or most likely category.
    
    Args:
        learning_outcome: The learning outcome text
        client: OpenAI client
    
    Returns:
        Single category name that best fits the learning outcome
    """
    binary_decisions = get_zero_shot_prediction_binary_class(learning_outcome, client)
    
    # Find categories marked as 1
    selected_categories = [cat for cat, decision in binary_decisions.items() if decision == 1]
    
    if selected_categories:
        # Return first selected category
        return selected_categories[0]
    else:
        # If no categories selected, default to 'understand'
        return 'understand'

# Alias for backward compatibility with multi-class name
def get_zero_shot_prediction_multi_class(learning_outcome: str, client) -> Dict[str, int]:
    """Alias for binary classification to maintain compatibility."""
    return get_zero_shot_prediction_binary_class(learning_outcome, client)

# Legacy function for backwards compatibility
def get_zero_shot_prediction(learning_outcome: str, client, category: str) -> Optional[bool]:
    """
    Legacy function for single category prediction.
    Now uses binary decisions internally.
    """
    binary_decisions = get_zero_shot_prediction_binary_class(learning_outcome, client)
    
    # Return True if this category is marked as 1
    return binary_decisions.get(category, 0) == 1
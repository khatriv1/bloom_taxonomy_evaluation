# bloom_taxonomy_evaluation/prompting/contrastive_cot.py

"""
Contrastive Chain of Thought prompting for Bloom taxonomy classification.
Binary version: Uses both positive and negative reasoning for binary decisions.
"""

import time
import json
import re
from typing import List, Optional, Dict
from utils.bloom_rubric import BloomRubric

def get_contrastive_cot_prediction_binary_class(learning_outcome: str, client) -> Dict[str, int]:
    """
    Get Contrastive Chain of Thought binary prediction for all 6 Bloom categories.
    
    Args:
        learning_outcome: The learning outcome text
        client: OpenAI client
    
    Returns:
        Dictionary with 1/0 decisions for all 6 categories
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

Now classify this learning outcome using both positive and negative reasoning:

Learning Outcome: "{learning_outcome}"

INSTRUCTIONS:
1. For each category, consider why it MIGHT apply (positive reasoning)
2. For each category, consider why it might NOT apply (negative reasoning)
3. Based on both reasonings, make binary decisions (1 if applies, 0 if doesn't)
4. Multiple categories can be 1 if the outcome involves multiple types of thinking

Provide your analysis and final binary decisions in this JSON format:

{{
    "remember": 0,
    "understand": 0,
    "apply": 0,
    "analyze": 0,
    "evaluate": 0,
    "create": 0
}}

Analysis and Classification:"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert at classifying learning outcomes using Bloom's Taxonomy. Use contrastive reasoning and provide JSON binary decisions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=500
        )
        
        result = response.choices[0].message.content.strip()
        
        # Extract JSON from response
        binary_decisions = parse_binary_decisions(result, categories)
        
        return binary_decisions
        
    except Exception as e:
        print(f"Error in contrastive CoT binary classification: {str(e)}")
        return {cat: 0 for cat in categories}

def get_contrastive_cot_prediction_single_category(learning_outcome: str, client) -> str:
    """Get single best category using contrastive reasoning."""
    binary_decisions = get_contrastive_cot_prediction_binary_class(learning_outcome, client)
    
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
def get_contrastive_cot_prediction_multi_class(learning_outcome: str, client) -> Dict[str, int]:
    """Alias for binary classification to maintain compatibility."""
    return get_contrastive_cot_prediction_binary_class(learning_outcome, client)

# Legacy function for backwards compatibility
def get_contrastive_examples(category: str) -> str:
    """Legacy function - now provides general contrastive guidance."""
    return "Use both positive and negative reasoning to determine which categories apply."

def get_contrastive_cot_prediction(learning_outcome: str, client, category: str) -> Optional[bool]:
    """
    Legacy function for single category prediction.
    Now uses binary decisions internally.
    """
    binary_decisions = get_contrastive_cot_prediction_binary_class(learning_outcome, client)
    
    # Return True if this category is marked as 1
    return binary_decisions.get(category, 0) == 1
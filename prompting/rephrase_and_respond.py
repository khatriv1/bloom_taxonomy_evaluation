# bloom_taxonomy_evaluation/prompting/rephrase_and_respond.py

"""
Rephrase and Respond prompting for Bloom taxonomy classification.
Binary version: Rephrases outcome then provides binary decisions.
"""

import time
import json
import re
from typing import List, Optional, Tuple, Dict
from utils.bloom_rubric import BloomRubric

def rephrase_learning_outcome(learning_outcome: str, client) -> Optional[str]:
    """Rephrase the learning outcome to clarify its cognitive demands."""
    
    prompt = f"""Rephrase the following learning outcome to make its cognitive demands and intent clearer, while preserving all important information:

Original learning outcome: "{learning_outcome}"

Instructions:
1. Make the action verbs more explicit
2. Clarify what cognitive processes students need to use
3. Preserve the original meaning and scope
4. Make it easier to understand what students will actually do

Rephrased learning outcome:"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert at clarifying educational learning outcomes. Rephrase to make cognitive demands clear while preserving all information."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=150
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Error rephrasing learning outcome: {str(e)}")
        return None

def get_rephrase_respond_prediction_binary_class(learning_outcome: str, client) -> Dict[str, int]:
    """
    Get Rephrase and Respond binary prediction for all 6 Bloom categories.
    
    Args:
        learning_outcome: The learning outcome text
        client: OpenAI client
    
    Returns:
        Dictionary with 1/0 decisions for all 6 categories
    """
    categories = ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']
    
    rubric = BloomRubric()
    category_definitions = rubric.get_category_definitions()
    
    # Step 1: Rephrase the learning outcome
    rephrased = rephrase_learning_outcome(learning_outcome, client)
    if rephrased is None:
        rephrased = learning_outcome
    
    definitions_text = ""
    for cat in categories:
        definitions_text += f"{cat.upper()}: {category_definitions[cat]['description']}\n"

    # Step 2: Make binary decisions based on both versions
    prompt = f"""You are an expert educator classifying learning outcomes according to Bloom's Taxonomy.

BLOOM'S TAXONOMY CATEGORIES:
{definitions_text}

TASK: Analyze both versions of this learning outcome and decide which Bloom taxonomy categories apply. For each category, respond with 1 if it applies or 0 if it doesn't apply.

Original learning outcome: "{learning_outcome}"
Rephrased for clarity: "{rephrased}"

INSTRUCTIONS:
1. Consider both the original and rephrased versions
2. Identify the cognitive demands required
3. For each category, decide if it applies (1) or doesn't apply (0)
4. Multiple categories can be 1 if the outcome involves multiple types of thinking
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
                {"role": "system", "content": "You are an expert at classifying learning outcomes using Bloom's Taxonomy. Consider both versions and provide JSON binary decisions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=250
        )
        
        result = response.choices[0].message.content.strip()
        
        # Extract JSON from response
        binary_decisions = parse_binary_decisions(result, categories)
        
        return binary_decisions
        
    except Exception as e:
        print(f"Error in rephrase-respond binary classification: {str(e)}")
        return {cat: 0 for cat in categories}

def get_rephrase_respond_prediction_single_category(learning_outcome: str, client) -> str:
    """Get single best category using rephrase-respond approach."""
    binary_decisions = get_rephrase_respond_prediction_binary_class(learning_outcome, client)
    
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
def get_rephrase_respond_prediction_multi_class(learning_outcome: str, client) -> Dict[str, int]:
    """Alias for binary classification to maintain compatibility."""
    return get_rephrase_respond_prediction_binary_class(learning_outcome, client)

# Legacy function for backwards compatibility
def get_rephrase_respond_prediction(learning_outcome: str, client, category: str) -> Optional[bool]:
    """
    Legacy function for single category prediction.
    Now uses binary decisions internally.
    """
    binary_decisions = get_rephrase_respond_prediction_binary_class(learning_outcome, client)
    
    # Return True if this category is marked as 1
    return binary_decisions.get(category, 0) == 1
# bloom_taxonomy_evaluation/prompting/take_a_step_back.py

"""
Take a Step Back prompting for Bloom taxonomy classification.
Binary version: Derives principles then applies for binary decisions.
"""

import time
import json
import re
from typing import List, Optional, Dict
from utils.bloom_rubric import BloomRubric

def derive_classification_principles(client) -> Optional[str]:
    """Derive high-level principles for Bloom taxonomy classification."""
    
    prompt = f"""Take a step back and think about the fundamental principles for classifying learning outcomes according to Bloom's Taxonomy with binary decisions.

What are the key characteristics, patterns, and principles that help determine which of the 6 Bloom taxonomy levels apply to a learning outcome?

Consider:
1. What types of action verbs typically indicate each level?
2. How does cognitive complexity increase from Remember to Create?
3. How can you tell when multiple categories apply to one outcome?
4. What are common misconceptions when classifying learning outcomes?

List 5-7 high-level principles for binary Bloom taxonomy classification:"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing patterns in educational learning outcomes. Derive clear principles for binary Bloom taxonomy classification."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Error deriving principles: {str(e)}")
        return None

def get_take_step_back_prediction_binary_class(learning_outcome: str, client) -> Dict[str, int]:
    """
    Get Take a Step Back binary prediction for all 6 Bloom categories.
    
    Args:
        learning_outcome: The learning outcome text
        client: OpenAI client
    
    Returns:
        Dictionary with 1/0 decisions for all 6 categories
    """
    categories = ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']
    
    rubric = BloomRubric()
    category_definitions = rubric.get_category_definitions()
    
    # Step 1: Derive high-level principles
    principles = derive_classification_principles(client)
    if principles is None:
        principles = "Consider the action verbs and cognitive complexity required by the learning outcome."
    
    definitions_text = ""
    for cat in categories:
        definitions_text += f"{cat.upper()}: {category_definitions[cat]['description']}\n"

    # Step 2: Apply principles to make binary decisions
    prompt = f"""You are an expert educator classifying learning outcomes according to Bloom's Taxonomy.

BLOOM'S TAXONOMY CATEGORIES:
{definitions_text}

HIGH-LEVEL PRINCIPLES FOR BLOOM CLASSIFICATION:
{principles}

TASK: Apply these principles to make binary decisions for this learning outcome.

Learning Outcome: "{learning_outcome}"

INSTRUCTIONS:
1. Apply the general principles above to this specific case
2. Consider which categories apply to this outcome
3. Make binary decisions (1 if applies, 0 if doesn't) for each category
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
                {"role": "system", "content": "You are an expert at classifying learning outcomes using Bloom's Taxonomy. Apply the given principles and provide JSON binary decisions."},
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
        print(f"Error in take-step-back binary classification: {str(e)}")
        return {cat: 0 for cat in categories}

def get_take_step_back_prediction_single_category(learning_outcome: str, client) -> str:
    """Get single best category using step-back reasoning."""
    binary_decisions = get_take_step_back_prediction_binary_class(learning_outcome, client)
    
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
def get_take_step_back_prediction_multi_class(learning_outcome: str, client) -> Dict[str, int]:
    """Alias for binary classification to maintain compatibility."""
    return get_take_step_back_prediction_binary_class(learning_outcome, client)

# Legacy function for backwards compatibility
def get_take_step_back_prediction(learning_outcome: str, client, category: str) -> Optional[bool]:
    """
    Legacy function for single category prediction.
    Now uses binary decisions internally.
    """
    binary_decisions = get_take_step_back_prediction_binary_class(learning_outcome, client)
    
    # Return True if this category is marked as 1
    return binary_decisions.get(category, 0) == 1
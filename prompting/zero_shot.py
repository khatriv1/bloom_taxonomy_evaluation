# bloom_taxonomy_evaluation/prompting/zero_shot.py

"""
Zero-shot prompting for Bloom taxonomy classification.
Classifies learning outcomes into Bloom's 6 categories without examples.
"""

import time
from typing import Optional
from utils.bloom_rubric import BloomRubric

def get_zero_shot_prediction(learning_outcome: str, 
                           client,
                           category: str) -> Optional[bool]:
    """
    Get zero-shot prediction for a single category using Bloom rubric.
    
    Args:
        learning_outcome: The learning outcome text
        client: OpenAI client
        category: Category to classify for
    
    Returns:
        Boolean indicating if outcome belongs to category, None if failed
    """
    rubric = BloomRubric()
    prompt_descriptions = rubric.get_prompt_descriptions()
    
    if category not in prompt_descriptions:
        raise ValueError(f"Unknown category: {category}")
    
    # Create zero-shot prompt
    prompt = f"""Consider a learning outcome from an educational course below:

Learning Outcome: {learning_outcome}

If the statement below is true, please respond "true"; otherwise, please respond "false":

{prompt_descriptions[category]}"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing learning outcomes according to Bloom's Taxonomy. Respond only with 'true' or 'false'."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=10
        )
        
        result = response.choices[0].message.content.strip().lower()
        
        if result == "true":
            return True
        elif result == "false":
            return False
        else:
            print(f"Unexpected response for {category}: {result}")
            return None
            
    except Exception as e:
        print(f"Error getting prediction for {category}: {str(e)}")
        return None


def get_zero_shot_prediction_single_category(learning_outcome: str,
                                            client) -> str:
    """
    Get zero-shot prediction for the single best Bloom category.
    
    Args:
        learning_outcome: The learning outcome text
        client: OpenAI client
    
    Returns:
        Single category name that best fits the learning outcome
    """
    categories = [
        'remember', 'understand', 'apply', 'analyze', 'evaluate', 'create'
    ]
    
    rubric = BloomRubric()
    category_definitions = rubric.get_category_definitions()
    
    # Create comprehensive prompt for single category selection
    definitions_text = ""
    for cat in categories:
        definitions_text += f"\n{cat.upper()}: {category_definitions[cat]['description']}\n"
        definitions_text += f"Examples: {', '.join(category_definitions[cat]['examples'][:2])}\n"
    
    prompt = f"""You are an expert educator classifying learning outcomes according to Bloom's Taxonomy.

BLOOM'S TAXONOMY CATEGORIES:
{definitions_text}

TASK: Classify the following learning outcome into exactly ONE Bloom taxonomy category.

Learning Outcome: "{learning_outcome}"

INSTRUCTIONS:
1. Analyze the learning outcome carefully
2. Identify the primary cognitive demand required
3. Match to the most appropriate Bloom taxonomy level
4. Respond with only the category name: remember, understand, apply, analyze, evaluate, or create

Classification:"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert at classifying learning outcomes using Bloom's Taxonomy. Respond only with the category name."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=20
        )
        
        result = response.choices[0].message.content.strip().lower()
        
        # Validate and clean the response
        valid_categories = ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']
        
        for category in valid_categories:
            if category in result:
                return category
                
        # If no valid category found, return default
        print(f"Could not extract valid category from: {result}")
        return 'understand'  # Default fallback
        
    except Exception as e:
        print(f"Error in zero-shot classification: {str(e)}")
        return 'understand'  # Default fallback
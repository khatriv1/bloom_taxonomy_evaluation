# bloom_taxonomy_evaluation/prompting/cot.py

"""
Chain of Thought prompting for Bloom taxonomy classification.
Uses step-by-step reasoning before making classification decisions.
"""

import time
from typing import Optional
from utils.bloom_rubric import BloomRubric

def get_cot_prediction(learning_outcome: str,
                      client,
                      category: str) -> Optional[bool]:
    """
    Get Chain of Thought prediction for a single category.
    
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
    
    # Create Chain of Thought prompt with reasoning steps
    prompt = f"""Consider a learning outcome from an educational course below:

Learning Outcome: {learning_outcome}

Please analyze this learning outcome step by step to determine if it fits this category:

Category: {category}
Definition: {prompt_descriptions[category]}

Step 1: What is the main action verb or cognitive demand in this learning outcome?
Step 2: What type of cognitive process does this require from students?
Step 3: Does this match the definition of the {category} category?
Step 4: What specific words or phrases support your decision?

Based on your step-by-step analysis, respond with "true" if the learning outcome belongs to the {category} category, or "false" if it does not."""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing learning outcomes using Bloom's Taxonomy. Think step by step and end your response with either 'true' or 'false'."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=200
        )
        
        result = response.choices[0].message.content.strip().lower()
        
        # Extract the final true/false from the reasoning
        if "true" in result.split()[-5:]:  # Check last few words
            return True
        elif "false" in result.split()[-5:]:
            return False
        else:
            print(f"Could not extract true/false from CoT response for {category}: {result}")
            return None
            
    except Exception as e:
        print(f"Error getting CoT prediction for {category}: {str(e)}")
        return None


def get_cot_prediction_single_category(learning_outcome: str,
                                     client) -> str:
    """
    Get Chain of Thought prediction for the single best Bloom category.
    
    Args:
        learning_outcome: The learning outcome text
        client: OpenAI client
    
    Returns:
        Single category name that best fits the learning outcome
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

TASK: Classify the following learning outcome into ONE primary Bloom taxonomy category using step-by-step reasoning.

Learning Outcome: "{learning_outcome}"

INSTRUCTIONS:
Think step-by-step about this learning outcome:

Step 1 - Identify key action verbs:
Step 2 - Analyze cognitive demands:
Step 3 - Compare to Bloom categories:
Step 4 - Determine best match:

Based on your step-by-step analysis, respond with only the category name: remember, understand, apply, analyze, evaluate, or create

Final Classification:"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert at classifying learning outcomes using Bloom's Taxonomy. Think step by step and end with the category name."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=300
        )
        
        result = response.choices[0].message.content.strip().lower()
        
        # Extract the final classification from the response
        lines = result.split('\n')
        for line in lines:
            if 'final classification' in line.lower() or 'classification:' in line.lower():
                # Extract the category from the line
                valid_categories = ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']
                for category in valid_categories:
                    if category in line.lower():
                        return category
        
        # If no clear final classification, look for categories in the entire response
        valid_categories = ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']
        for category in valid_categories:
            if category in result:
                return category
                
        # If no valid category found, return default
        print(f"Could not extract valid category from CoT response: {result}")
        return 'understand'  # Default fallback
        
    except Exception as e:
        print(f"Error in CoT classification: {str(e)}")
        return 'understand'  # Default fallback
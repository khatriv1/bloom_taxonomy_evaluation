# bloom_taxonomy_evaluation/prompting/rephrase_and_respond.py

"""
Rephrase and Respond prompting for Bloom taxonomy classification.
First rephrases the learning outcome to clarify meaning, then classifies.
"""

import time
from typing import List, Optional, Tuple
from utils.bloom_rubric import BloomRubric

def rephrase_learning_outcome(learning_outcome: str, client) -> Optional[str]:
    """
    Rephrase the learning outcome to clarify its cognitive demands.
    
    Args:
        learning_outcome: The original learning outcome
        client: OpenAI client
    
    Returns:
        Rephrased learning outcome or None if failed
    """
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

def get_rephrase_respond_prediction_single_category(learning_outcome: str,
                                                   client) -> str:
    """
    Get Rephrase and Respond prediction for the single best Bloom category.
    
    Args:
        learning_outcome: The learning outcome text
        client: OpenAI client
    
    Returns:
        Single category name that best fits the learning outcome
    """
    categories = ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']
    
    rubric = BloomRubric()
    category_definitions = rubric.get_category_definitions()
    
    # Step 1: Rephrase the learning outcome
    rephrased = rephrase_learning_outcome(learning_outcome, client)
    if rephrased is None:
        print(f"Failed to rephrase learning outcome, using original")
        rephrased = learning_outcome
    
    definitions_text = ""
    for cat in categories:
        definitions_text += f"{cat.upper()}: {category_definitions[cat]['description']}\n"

    # Step 2: Create classification prompt with both original and rephrased
    prompt = f"""You are an expert educator classifying learning outcomes according to Bloom's Taxonomy.

BLOOM'S TAXONOMY CATEGORIES:
{definitions_text}

TASK: Classify this learning outcome into ONE primary Bloom taxonomy category.

Original learning outcome: "{learning_outcome}"
Rephrased for clarity: "{rephrased}"

INSTRUCTIONS:
1. Consider both the original and rephrased versions
2. Identify the primary cognitive demand required
3. Focus on what students actually need to DO cognitively
4. Match to the most appropriate Bloom taxonomy level
5. Respond with only the category name: remember, understand, apply, analyze, evaluate, or create

Analysis:
Based on both versions, the primary cognitive demand is...

Classification:"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert at classifying learning outcomes using Bloom's Taxonomy. Consider both versions to make accurate classifications. End with the category name."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=200
        )
        
        result = response.choices[0].message.content.strip().lower()
        
        # Extract classification
        lines = result.split('\n')
        for line in lines:
            if 'classification:' in line.lower():
                valid_categories = ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']
                for category in valid_categories:
                    if category in line.lower():
                        return category
        
        # If no explicit classification line, search entire response
        valid_categories = ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']
        for category in valid_categories:
            if category in result:
                return category
                
        # If no valid category found, return default
        print(f"Could not extract valid category from Rephrase-Respond response: {result}")
        return 'understand'  # Default fallback
        
    except Exception as e:
        print(f"Error in rephrase-respond classification: {str(e)}")
        return 'understand'  # Default fallback
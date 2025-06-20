# bloom_taxonomy_evaluation/prompting/take_a_step_back.py

"""
Take a Step Back prompting for Bloom taxonomy classification.
First derives high-level principles, then applies them to classification.
"""

import time
from typing import List, Optional, Dict
from utils.bloom_rubric import BloomRubric

def derive_classification_principles(client) -> Optional[str]:
    """
    Derive high-level principles for Bloom taxonomy classification.
    
    Args:
        client: OpenAI client
    
    Returns:
        String containing derived principles or None if failed
    """
    
    prompt = f"""Take a step back and think about the fundamental principles for classifying learning outcomes according to Bloom's Taxonomy.

What are the key characteristics, patterns, and principles that help distinguish between the 6 Bloom taxonomy levels when analyzing learning outcomes?

Consider:
1. What types of action verbs typically indicate each level?
2. What cognitive processes distinguish each level from others?
3. How does complexity increase from Remember to Create?
4. What are common misconceptions when classifying learning outcomes?

List 5-7 high-level principles for Bloom taxonomy classification:"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing patterns in educational learning outcomes. Derive clear, high-level principles for Bloom taxonomy classification."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Error deriving principles for Bloom classification: {str(e)}")
        return None

def get_take_step_back_prediction_single_category(learning_outcome: str,
                                                 client) -> str:
    """
    Get Take a Step Back prediction for the single best Bloom category.
    
    Args:
        learning_outcome: The learning outcome text
        client: OpenAI client
    
    Returns:
        Single category name that best fits the learning outcome
    """
    categories = ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']
    
    rubric = BloomRubric()
    category_definitions = rubric.get_category_definitions()
    
    # Step 1: Derive high-level principles
    principles = derive_classification_principles(client)
    if principles is None:
        print(f"Failed to derive principles for Bloom classification")
        principles = "Consider the action verbs and cognitive complexity required by the learning outcome."
    
    definitions_text = ""
    for cat in categories:
        definitions_text += f"{cat.upper()}: {category_definitions[cat]['description']}\n"

    # Step 2: Apply principles to classify
    prompt = f"""You are an expert educator classifying learning outcomes according to Bloom's Taxonomy.

BLOOM'S TAXONOMY CATEGORIES:
{definitions_text}

HIGH-LEVEL PRINCIPLES FOR BLOOM CLASSIFICATION:
{principles}

TASK: Now apply these principles to classify this specific learning outcome.

Learning Outcome: "{learning_outcome}"

INSTRUCTIONS:
1. Apply the general principles above to this specific case
2. Identify the primary cognitive demand required
3. Determine which Bloom taxonomy level best matches
4. Respond with only the category name: remember, understand, apply, analyze, evaluate, or create

Based on the principles above, this learning outcome primarily requires students to...

Classification:"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert at classifying learning outcomes using Bloom's Taxonomy. Apply the given principles to make accurate classifications. End with the category name."},
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
        
        # If no explicit classification, search entire response
        valid_categories = ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']
        for category in valid_categories:
            if category in result:
                return category
                
        # If no valid category found, return default
        print(f"Could not extract valid category from Take-Step-Back response: {result}")
        return 'understand'  # Default fallback
        
    except Exception as e:
        print(f"Error in take-step-back classification: {str(e)}")
        return 'understand'  # Default fallback
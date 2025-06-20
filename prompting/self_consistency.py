# bloom_taxonomy_evaluation/prompting/self_consistency.py

"""
Self-Consistency prompting for Bloom taxonomy classification.
Samples multiple reasoning paths and takes majority vote.
"""

import time
from typing import List, Optional, Dict
from collections import Counter
from utils.bloom_rubric import BloomRubric

def get_single_reasoning_path(learning_outcome: str,
                            client,
                            temperature: float = 0.7) -> Optional[str]:
    """
    Get a single reasoning path for classification.
    
    Args:
        learning_outcome: The learning outcome text
        client: OpenAI client
        temperature: Sampling temperature for diversity
    
    Returns:
        String category prediction or None if failed
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

Think through this step-by-step and explain your reasoning:
1. What is the main action verb or cognitive demand in this outcome?
2. What cognitive processes does this require from students?
3. How does this match the Bloom taxonomy categories?
4. What is the primary cognitive level involved?

Based on your analysis, classify this learning outcome into ONE category: remember, understand, apply, analyze, evaluate, or create

Classification:"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert at classifying learning outcomes using Bloom's Taxonomy. Provide reasoning and end with the category name."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=250
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
                
        return None
            
    except Exception as e:
        print(f"Error in reasoning path: {str(e)}")
        return None

def get_self_consistency_prediction_single_category(learning_outcome: str,
                                                   client,
                                                   n_samples: int = 5) -> str:
    """
    Get Self-Consistency prediction using multiple reasoning paths.
    
    Args:
        learning_outcome: The learning outcome text
        client: OpenAI client
        n_samples: Number of reasoning paths to sample
    
    Returns:
        Single category name that best fits the learning outcome
    """
    
    # Collect predictions from multiple reasoning paths
    predictions = []
    
    for i in range(n_samples):
        # Vary temperature for diversity
        temp = 0.5 + (i * 0.1)  # 0.5, 0.6, 0.7, 0.8, 0.9
        
        prediction = get_single_reasoning_path(learning_outcome, client, temp)
        
        if prediction is not None:
            predictions.append(prediction)
        
        # Small delay between samples
        time.sleep(0.2)
    
    if not predictions:
        print(f"No valid predictions obtained for self-consistency")
        return 'understand'  # Default fallback
    
    # Take majority vote
    vote_counts = Counter(predictions)
    majority_vote = vote_counts.most_common(1)[0][0]
    confidence = vote_counts[majority_vote] / len(predictions)
    
    print(f"Self-consistency votes: {dict(vote_counts)}, confidence: {confidence:.2f}")
    
    return majority_vote
# bloom_taxonomy_evaluation/prompting/auto_cot.py

"""
Auto-CoT (Automatic Chain of Thought) prompting for Bloom taxonomy classification.
Automatically generates reasoning chains for classification.
"""

import time
from typing import List, Optional, Dict
from utils.bloom_rubric import BloomRubric

def generate_auto_cot_examples(category: str, client) -> str:
    """Generate automatic reasoning chains for examples."""
    
    sample_outcomes = {
        'remember': "Students will list the major presidents of the United States",
        'understand': "Students will explain the process of photosynthesis in plants", 
        'apply': "Students will solve linear equations using algebraic methods",
        'analyze': "Students will compare different forms of government",
        'evaluate': "Students will assess the quality of research studies",
        'create': "Students will design an original science experiment"
    }
    
    outcome = sample_outcomes.get(category, "Students will demonstrate knowledge")
    
    prompt = f"""You are an expert educator. Create a detailed step-by-step reasoning example for classifying learning outcomes according to Bloom's Taxonomy.

Learning Outcome: "{outcome}"

Create a step-by-step reasoning example that shows how to classify this learning outcome:

Step 1 - Identify key verbs:
Step 2 - Analyze cognitive demand:  
Step 3 - Match to Bloom level:
Step 4 - Final classification:

Provide a complete reasoning example."""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.3
        )
        
        reasoning = response.choices[0].message.content.strip()
        return f"Learning Outcome: \"{outcome}\"\n{reasoning}\n"
        
    except Exception as e:
        print(f"Error generating auto-CoT example: {e}")
        return f"Learning Outcome: \"{outcome}\"\nStep 1: Analyze the verbs\nStep 2: Determine cognitive complexity\nStep 3: Match to category\nStep 4: {category}\n"

def get_auto_cot_prediction_single_category(learning_outcome: str,
                                           client) -> str:
    """
    Get Auto Chain of Thought prediction for the single best Bloom category.
    
    Args:
        learning_outcome: The learning outcome text
        client: OpenAI client
    
    Returns:
        Single category name that best fits the learning outcome
    """
    categories = ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']
    
    rubric = BloomRubric()
    category_definitions = rubric.get_category_definitions()
    
    # Generate automatic CoT examples for 2 categories
    demo_categories = ['remember', 'apply']  # Use 2 examples to avoid token limits
    auto_examples = []
    
    for demo_cat in demo_categories:
        example = generate_auto_cot_examples(demo_cat, client)
        auto_examples.append(example)
        time.sleep(0.3)  # Rate limiting
    
    examples_text = "\n".join(auto_examples)
    
    definitions_text = ""
    for cat in categories:
        definitions_text += f"{cat.upper()}: {category_definitions[cat]['description']}\n"

    prompt = f"""You are an expert educator classifying learning outcomes according to Bloom's Taxonomy.

BLOOM'S TAXONOMY CATEGORIES:
{definitions_text}

Here are examples of step-by-step reasoning for Bloom taxonomy classification:

{examples_text}

Now, using the same step-by-step approach, classify this learning outcome:

Learning Outcome: "{learning_outcome}"

Step 1 - Identify key verbs:
Step 2 - Analyze cognitive demand:
Step 3 - Match to Bloom level:
Step 4 - Final classification:"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert at classifying learning outcomes using Bloom's Taxonomy. Follow the step-by-step approach and end with the category name."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=300
        )
        
        result = response.choices[0].message.content.strip().lower()
        
        # Extract the final classification
        lines = result.split('\n')
        for line in lines:
            if 'step 4' in line.lower() or 'final classification' in line.lower():
                valid_categories = ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']
                for category in valid_categories:
                    if category in line.lower():
                        return category
        
        # If no clear final classification in Step 4, search entire response
        valid_categories = ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']
        for category in valid_categories:
            if category in result:
                return category
                
        # If no valid category found, return default
        print(f"Could not extract valid category from Auto-CoT response: {result}")
        return 'understand'  # Default fallback
        
    except Exception as e:
        print(f"Error in auto-CoT classification: {str(e)}")
        return 'understand'  # Default fallback
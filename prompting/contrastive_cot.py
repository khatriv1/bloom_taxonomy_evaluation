# bloom_taxonomy_evaluation/prompting/contrastive_cot.py

"""
Contrastive Chain of Thought prompting for Bloom taxonomy classification.
Uses both positive and negative reasoning to improve classification.
"""

import time
from typing import List, Optional, Dict
from utils.bloom_rubric import BloomRubric

def get_contrastive_examples(category: str) -> str:
    """Generate contrastive reasoning examples for each category."""
    contrastive_examples = {
        'remember': {
            "positive": {
                "outcome": "Students will memorize the periodic table elements",
                "positive_reasoning": "This requires students to retrieve factual information from memory without modification or understanding.",
                "negative_reasoning": "This is NOT about understanding concepts (no explanation required), NOT about applying knowledge (no problem-solving), NOT about analysis (no breaking down).",
                "answer": "remember"
            },
            "negative": {
                "outcome": "Students will explain how photosynthesis works",
                "positive_reasoning": "This could seem like recalling facts about photosynthesis.",
                "negative_reasoning": "However, 'explain' requires comprehension and interpretation of meaning, NOT just retrieval from memory. This is understanding.",
                "answer": "understand"
            }
        },
        'understand': {
            "positive": {
                "outcome": "Students will describe the water cycle process",
                "positive_reasoning": "This requires comprehension and the ability to express meaning in their own words about natural processes.",
                "negative_reasoning": "This is NOT just memorization (no rote recall), NOT application (no procedure execution), NOT analysis (no breaking into parts).",
                "answer": "understand"
            },
            "negative": {
                "outcome": "Students will solve math problems using formulas",
                "positive_reasoning": "This mentions mathematical processes.",
                "negative_reasoning": "However, 'solve using formulas' requires executing procedures in specific situations, NOT just understanding concepts. This is application.",
                "answer": "apply"
            }
        },
        'apply': {
            "positive": {
                "outcome": "Students will calculate areas using geometric formulas",
                "positive_reasoning": "This requires carrying out established procedures (formulas) in specific problem-solving situations.",
                "negative_reasoning": "This is NOT just understanding formulas (no explanation), NOT analysis (no comparison), NOT creation (using existing methods).",
                "answer": "apply"
            },
            "negative": {
                "outcome": "Students will compare different geometric shapes",
                "positive_reasoning": "This involves geometric knowledge.",
                "negative_reasoning": "However, 'compare' requires breaking down and examining relationships between elements, NOT just applying procedures. This is analysis.",
                "answer": "analyze"
            }
        }
    }
    
    # Get examples for the category
    if category not in contrastive_examples:
        return "No contrastive examples available for this category."
    
    examples = contrastive_examples[category]
    
    # Format contrastive examples
    formatted = []
    for example_type, example in examples.items():
        formatted.append(f"Example ({example_type}):\nLearning Outcome: {example['outcome']}\nPositive reasoning: {example['positive_reasoning']}\nNegative reasoning: {example['negative_reasoning']}\nCorrect Classification: {example['answer']}")
    
    return "\n\n".join(formatted)

def get_contrastive_cot_prediction_single_category(learning_outcome: str,
                                                  client) -> str:
    """
    Get Contrastive Chain of Thought prediction for the single best Bloom category.
    
    Args:
        learning_outcome: The learning outcome text
        client: OpenAI client
    
    Returns:
        Single category name that best fits the learning outcome
    """
    categories = ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']
    
    rubric = BloomRubric()
    category_definitions = rubric.get_category_definitions()
    
    # Get contrastive examples for key categories
    remember_examples = get_contrastive_examples('remember')
    understand_examples = get_contrastive_examples('understand')
    apply_examples = get_contrastive_examples('apply')
    
    definitions_text = ""
    for cat in categories:
        definitions_text += f"{cat.upper()}: {category_definitions[cat]['description']}\n"

    prompt = f"""You are an expert educator classifying learning outcomes according to Bloom's Taxonomy.

BLOOM'S TAXONOMY CATEGORIES:
{definitions_text}

Here are contrastive examples showing both why outcomes do and don't belong to certain categories:

REMEMBER Examples:
{remember_examples}

UNDERSTAND Examples:
{understand_examples}

APPLY Examples:
{apply_examples}

Now classify this learning outcome using both positive and negative reasoning:

Learning Outcome: "{learning_outcome}"

INSTRUCTIONS:
1. First, provide positive reasoning (why it MIGHT belong to each category)
2. Then, provide negative reasoning (why it might NOT belong to each category)
3. Finally, determine the best classification based on both reasonings
4. Respond with only the category name: remember, understand, apply, analyze, evaluate, or create

Analysis:

Final Classification:"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert at classifying learning outcomes using Bloom's Taxonomy. Use contrastive reasoning and end with the category name."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=400
        )
        
        result = response.choices[0].message.content.strip().lower()
        
        # Extract final classification
        lines = result.split('\n')
        for line in lines:
            if 'final classification' in line.lower():
                valid_categories = ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']
                for category in valid_categories:
                    if category in line.lower():
                        return category
        
        # If no explicit final classification, search entire response
        valid_categories = ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']
        for category in valid_categories:
            if category in result:
                return category
                
        # If no valid category found, return default
        print(f"Could not extract valid category from Contrastive CoT response: {result}")
        return 'understand'  # Default fallback
        
    except Exception as e:
        print(f"Error in contrastive CoT classification: {str(e)}")
        return 'understand'  # Default fallback
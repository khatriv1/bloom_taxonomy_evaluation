# bloom_taxonomy_evaluation/prompting/few_shot.py

"""
Few-shot prompting for Bloom taxonomy classification.
Provides examples before asking for classification.
"""

import time
from typing import Optional
from utils.bloom_rubric import BloomRubric

def get_few_shot_examples(category: str) -> str:
    """Get few-shot examples for each category."""
    examples = {
        'remember': [
            ("Students will list the major battles of World War II", "true"),
            ("Students will identify the parts of a cell", "true"),
            ("Students will analyze the causes of poverty", "false")
        ],
        'understand': [
            ("Students will explain the water cycle process", "true"),
            ("Students will describe the main themes in the novel", "true"),
            ("Students will memorize multiplication tables", "false")
        ],
        'apply': [
            ("Students will solve quadratic equations using formulas", "true"),
            ("Students will demonstrate proper laboratory procedures", "true"),
            ("Students will list the periodic table elements", "false")
        ],
        'analyze': [
            ("Students will compare different economic systems", "true"),
            ("Students will examine the causes of climate change", "true"),
            ("Students will recite the Pledge of Allegiance", "false")
        ],
        'evaluate': [
            ("Students will assess the credibility of news sources", "true"),
            ("Students will critique different research methodologies", "true"),
            ("Students will explain photosynthesis", "false")
        ],
        'create': [
            ("Students will design an original experiment", "true"),
            ("Students will compose an original poem", "true"),
            ("Students will calculate the area of a triangle", "false")
        ]
    }
    
    # Format examples for prompt
    formatted_examples = []
    for outcome, label in examples.get(category, []):
        formatted_examples.append(f"Learning Outcome: {outcome}\nAnswer: {label}")
    
    return "\n\n".join(formatted_examples)

def get_few_shot_prediction(learning_outcome: str,
                          client,
                          category: str) -> Optional[bool]:
    """
    Get few-shot prediction for a single category.
    
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
    
    # Get examples for this category
    examples = get_few_shot_examples(category)
    
    # Create few-shot prompt
    prompt = f"""You are classifying learning outcomes according to Bloom's Taxonomy.

Category: {category}
Definition: {prompt_descriptions[category]}

Here are some examples:

{examples}

Now classify this learning outcome:

Learning Outcome: {learning_outcome}

Answer:"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing learning outcomes using Bloom's Taxonomy. Respond only with 'true' or 'false'."},
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
        print(f"Error getting few-shot prediction for {category}: {str(e)}")
        return None


def get_few_shot_prediction_single_category(learning_outcome: str,
                                           client) -> str:
    """
    Get few-shot prediction for the single best Bloom category.
    
    Args:
        learning_outcome: The learning outcome text
        client: OpenAI client
    
    Returns:
        Single category name that best fits the learning outcome
    """
    categories = ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']
    
    # Create examples for all categories
    all_examples = """
EXAMPLES:

Learning Outcome: "Students will memorize the periodic table of elements"
Classification: remember
Explanation: This requires retrieving factual information from memory.

Learning Outcome: "Students will explain the process of photosynthesis"
Classification: understand
Explanation: This requires comprehending and expressing the meaning of biological processes.

Learning Outcome: "Students will use statistical formulas to solve probability problems"
Classification: apply
Explanation: This requires carrying out procedures in specific situations.

Learning Outcome: "Students will compare and contrast different economic theories"
Classification: analyze
Explanation: This requires breaking down concepts and examining relationships between parts.

Learning Outcome: "Students will assess the validity of research methodology in published studies"
Classification: evaluate
Explanation: This requires making judgments based on criteria and standards.

Learning Outcome: "Students will develop an original research proposal for their thesis"
Classification: create
Explanation: This requires putting elements together to form something novel and coherent.
"""

    rubric = BloomRubric()
    category_definitions = rubric.get_category_definitions()
    
    definitions_text = ""
    for cat in categories:
        definitions_text += f"{cat.upper()}: {category_definitions[cat]['description']}\n"

    prompt = f"""You are an expert educator classifying learning outcomes according to Bloom's Taxonomy.

BLOOM'S TAXONOMY CATEGORIES:
{definitions_text}

{all_examples}

TASK: Now classify the following learning outcome into ONE primary Bloom taxonomy category.

Learning Outcome: "{learning_outcome}"

INSTRUCTIONS:
1. Analyze the learning outcome carefully
2. Compare it to the examples provided
3. Determine which Bloom taxonomy level best describes the cognitive demand
4. Focus on the primary action verb and learning goal
5. Respond with only the category name: remember, understand, apply, analyze, evaluate, or create

Classification:"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert at classifying learning outcomes using Bloom's Taxonomy. Learn from the examples provided. Respond only with the category name."},
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
        print(f"Error in few-shot classification: {str(e)}")
        return 'understand'  # Default fallback
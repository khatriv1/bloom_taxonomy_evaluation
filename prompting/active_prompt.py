# bloom_taxonomy_evaluation/prompting/active_prompt.py

"""
Active prompting for Bloom taxonomy classification.
Iteratively selects informative examples to improve classification.
"""

import time
import numpy as np
from typing import List, Optional, Dict, Tuple
from utils.bloom_rubric import BloomRubric

class ActivePromptSelector:
    """Selects most informative examples for active learning."""
    
    def __init__(self):
        self.example_pool = self._initialize_example_pool()
        self.selected_examples = {category: [] for category in self.example_pool.keys()}
        self.uncertainty_scores = {}
    
    def _initialize_example_pool(self) -> Dict[str, List[Tuple[str, bool]]]:
        """Initialize pool of examples for active selection."""
        return {
            'remember': [
                ("Students will recall the names of all 50 US states", True),
                ("Students will list the major battles of World War II", True),
                ("Students will analyze complex social problems", False),
                ("Students will identify the parts of a microscope", True),
                ("Students will create original artwork", False)
            ],
            'understand': [
                ("Students will explain the water cycle process", True),
                ("Students will describe the main themes in literature", True),
                ("Students will memorize multiplication tables", False),
                ("Students will interpret data from graphs", True),
                ("Students will design a new experiment", False)
            ],
            'apply': [
                ("Students will solve quadratic equations using formulas", True),
                ("Students will demonstrate proper laboratory procedures", True),
                ("Students will list the periodic elements", False),
                ("Students will use statistical software to analyze data", True),
                ("Students will evaluate research quality", False)
            ],
            'analyze': [
                ("Students will compare different economic systems", True),
                ("Students will examine the causes of climate change", True),
                ("Students will recite poetry from memory", False),
                ("Students will break down complex arguments", True),
                ("Students will create new solutions", False)
            ],
            'evaluate': [
                ("Students will assess the credibility of news sources", True),
                ("Students will critique research methodologies", True),
                ("Students will explain basic concepts", False),
                ("Students will judge the effectiveness of policies", True),
                ("Students will apply known formulas", False)
            ],
            'create': [
                ("Students will design an original experiment", True),
                ("Students will compose an original poem", True),
                ("Students will calculate areas using formulas", False),
                ("Students will develop innovative business plans", True),
                ("Students will recall historical dates", False)
            ]
        }
    
    def select_examples(self, category: str, n_examples: int = 3) -> List[Tuple[str, str]]:
        """Select most informative examples using uncertainty sampling."""
        available_examples = self.example_pool[category]
        
        # For first iteration, select diverse examples
        if not self.selected_examples[category]:
            # Select one positive, one negative, and one uncertain
            positive = [ex for ex in available_examples if ex[1]]
            negative = [ex for ex in available_examples if not ex[1]]
            
            selected = []
            if positive:
                selected.append((positive[0][0], "true"))
            if negative:
                selected.append((negative[0][0], "false"))
            if len(positive) > 1:
                selected.append((positive[1][0], "true"))
            
            return selected[:n_examples]
        
        # For subsequent iterations, use uncertainty scores
        return self._select_by_uncertainty(category, n_examples)
    
    def _select_by_uncertainty(self, category: str, n_examples: int) -> List[Tuple[str, str]]:
        """Select examples with highest uncertainty."""
        examples = self.example_pool[category]
        selected = []
        
        for outcome, label in examples[:n_examples]:
            selected.append((outcome, "true" if label else "false"))
        
        return selected

def get_active_prompt_prediction_single_category(learning_outcome: str,
                                                client) -> str:
    """
    Get active prompting prediction for the single best Bloom category.
    
    Args:
        learning_outcome: The learning outcome text
        client: OpenAI client
    
    Returns:
        Single category name that best fits the learning outcome
    """
    categories = ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']
    
    rubric = BloomRubric()
    category_definitions = rubric.get_category_definitions()
    selector = ActivePromptSelector()
    
    # Get actively selected examples for each category
    examples_text = ""
    for cat in categories:
        selected_examples = selector.select_examples(cat, n_examples=2)
        examples_text += f"\n{cat.upper()} examples:\n"
        for ex_outcome, ex_label in selected_examples:
            examples_text += f"- \"{ex_outcome}\" → {ex_label}\n"
    
    definitions_text = ""
    for cat in categories:
        definitions_text += f"{cat.upper()}: {category_definitions[cat]['description']}\n"

    prompt = f"""You are an expert educator classifying learning outcomes according to Bloom's Taxonomy.

BLOOM'S TAXONOMY CATEGORIES:
{definitions_text}

CAREFULLY SELECTED EXAMPLES:
{examples_text}

TASK: Using the examples above, classify this learning outcome into ONE primary Bloom taxonomy category.

Learning Outcome: "{learning_outcome}"

INSTRUCTIONS:
1. Learn from the selected examples provided
2. Identify the primary cognitive demand in the learning outcome
3. Match to the most appropriate Bloom taxonomy level
4. Respond with only the category name: remember, understand, apply, analyze, evaluate, or create

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
        print(f"Error in active prompting classification: {str(e)}")
        return 'understand'  # Default fallback
# bloom_taxonomy_evaluation/prompting/active_prompt.py

"""
Active prompting for Bloom taxonomy classification.
Binary version: Iteratively selects informative examples for binary classification.
"""

import time
import json
import re
import numpy as np
from typing import List, Optional, Dict, Tuple
from utils.bloom_rubric import BloomRubric

class ActivePromptSelector:
    """Selects most informative examples for active learning."""
    
    def __init__(self):
        self.example_pool = self._initialize_example_pool()
        self.selected_examples = {category: [] for category in self.example_pool.keys()}
    
    def _initialize_example_pool(self) -> Dict[str, List[Tuple[str, Dict[str, int]]]]:
        """Initialize pool of examples for active selection with binary labels."""
        return {
            'remember': [
                ("Students will recall the names of all 50 US states", {"remember": 1, "understand": 0, "apply": 0, "analyze": 0, "evaluate": 0, "create": 0}),
                ("Students will list the major battles of World War II", {"remember": 1, "understand": 0, "apply": 0, "analyze": 0, "evaluate": 0, "create": 0}),
                ("Students will identify the parts of a microscope", {"remember": 1, "understand": 0, "apply": 0, "analyze": 0, "evaluate": 0, "create": 0}),
            ],
            'understand': [
                ("Students will explain the water cycle process", {"remember": 0, "understand": 1, "apply": 0, "analyze": 0, "evaluate": 0, "create": 0}),
                ("Students will describe the main themes in literature", {"remember": 0, "understand": 1, "apply": 0, "analyze": 0, "evaluate": 0, "create": 0}),
                ("Students will interpret data from graphs", {"remember": 0, "understand": 1, "apply": 0, "analyze": 0, "evaluate": 0, "create": 0}),
            ],
            'apply': [
                ("Students will solve quadratic equations using formulas", {"remember": 0, "understand": 0, "apply": 1, "analyze": 0, "evaluate": 0, "create": 0}),
                ("Students will demonstrate proper laboratory procedures", {"remember": 0, "understand": 0, "apply": 1, "analyze": 0, "evaluate": 0, "create": 0}),
                ("Students will use statistical software to analyze data", {"remember": 0, "understand": 0, "apply": 1, "analyze": 0, "evaluate": 0, "create": 0}),
            ]
        }
    
    def select_examples(self, category: str, n_examples: int = 3) -> List[Tuple[str, Dict[str, int]]]:
        """Select most informative examples."""
        available_examples = self.example_pool.get(category, [])
        return available_examples[:n_examples]

def get_active_prompt_prediction_binary_class(learning_outcome: str, client) -> Dict[str, int]:
    """
    Get active prompting binary prediction for all 6 Bloom categories.
    
    Args:
        learning_outcome: The learning outcome text
        client: OpenAI client
    
    Returns:
        Dictionary with 1/0 decisions for all 6 categories
    """
    categories = ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']
    
    rubric = BloomRubric()
    category_definitions = rubric.get_category_definitions()
    selector = ActivePromptSelector()
    
    # Get actively selected examples for key categories
    examples_text = ""
    for cat in ['remember', 'understand', 'apply']:  # Limit to avoid token issues
        selected_examples = selector.select_examples(cat, n_examples=2)
        examples_text += f"\n{cat.upper()} examples:\n"
        for ex_outcome, ex_binary in selected_examples:
            examples_text += f"- \"{ex_outcome}\" → {ex_binary}\n"
    
    definitions_text = ""
    for cat in categories:
        definitions_text += f"{cat.upper()}: {category_definitions[cat]['description']}\n"

    prompt = f"""You are an expert educator classifying learning outcomes according to Bloom's Taxonomy.

BLOOM'S TAXONOMY CATEGORIES:
{definitions_text}

CAREFULLY SELECTED EXAMPLES:
{examples_text}

TASK: Using the examples above, analyze this learning outcome and decide which Bloom taxonomy categories apply. For each category, respond with 1 if it applies or 0 if it doesn't apply.

Learning Outcome: "{learning_outcome}"

INSTRUCTIONS:
1. Learn from the selected examples provided
2. Identify the cognitive demands required
3. For each category, decide if it applies (1) or doesn't apply (0)
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
                {"role": "system", "content": "You are an expert at classifying learning outcomes using Bloom's Taxonomy. Learn from examples and provide JSON binary decisions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=200
        )
        
        result = response.choices[0].message.content.strip()
        
        # Extract JSON from response
        binary_decisions = parse_binary_decisions(result, categories)
        
        return binary_decisions
        
    except Exception as e:
        print(f"Error in active prompting binary classification: {str(e)}")
        return {cat: 0 for cat in categories}

def get_active_prompt_prediction_single_category(learning_outcome: str, client) -> str:
    """Get single best category using active prompting binary decisions."""
    binary_decisions = get_active_prompt_prediction_binary_class(learning_outcome, client)
    
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
def get_active_prompt_prediction_multi_class(learning_outcome: str, client) -> Dict[str, int]:
    """Alias for binary classification to maintain compatibility."""
    return get_active_prompt_prediction_binary_class(learning_outcome, client)

# =====================================
# bloom_taxonomy_evaluation/prompting/contrastive_cot.py

"""
Contrastive Chain of Thought prompting for Bloom taxonomy classification.
Binary version: Uses both positive and negative reasoning for binary decisions.
"""

import time
import json
import re
from typing import List, Optional, Dict
from utils.bloom_rubric import BloomRubric

def get_contrastive_cot_prediction_binary_class(learning_outcome: str, client) -> Dict[str, int]:
    """
    Get Contrastive Chain of Thought binary prediction for all 6 Bloom categories.
    
    Args:
        learning_outcome: The learning outcome text
        client: OpenAI client
    
    Returns:
        Dictionary with 1/0 decisions for all 6 categories
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

Now classify this learning outcome using both positive and negative reasoning:

Learning Outcome: "{learning_outcome}"

INSTRUCTIONS:
1. For each category, consider why it MIGHT apply (positive reasoning)
2. For each category, consider why it might NOT apply (negative reasoning)
3. Based on both reasonings, make binary decisions (1 if applies, 0 if doesn't)
4. Multiple categories can be 1 if the outcome involves multiple types of thinking

Provide your analysis and final binary decisions in this JSON format:

{{
    "remember": 0,
    "understand": 0,
    "apply": 0,
    "analyze": 0,
    "evaluate": 0,
    "create": 0
}}

Analysis and Classification:"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert at classifying learning outcomes using Bloom's Taxonomy. Use contrastive reasoning and provide JSON binary decisions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=500
        )
        
        result = response.choices[0].message.content.strip()
        
        # Extract JSON from response
        binary_decisions = parse_binary_decisions(result, categories)
        
        return binary_decisions
        
    except Exception as e:
        print(f"Error in contrastive CoT binary classification: {str(e)}")
        return {cat: 0 for cat in categories}

def get_contrastive_cot_prediction_single_category(learning_outcome: str, client) -> str:
    """Get single best category using contrastive reasoning."""
    binary_decisions = get_contrastive_cot_prediction_binary_class(learning_outcome, client)
    
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
def get_contrastive_cot_prediction_multi_class(learning_outcome: str, client) -> Dict[str, int]:
    """Alias for binary classification to maintain compatibility."""
    return get_contrastive_cot_prediction_binary_class(learning_outcome, client)

# =====================================
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

# =====================================
# bloom_taxonomy_evaluation/prompting/rephrase_and_respond.py

"""
Rephrase and Respond prompting for Bloom taxonomy classification.
Binary version: Rephrases outcome then provides binary decisions.
"""

import time
import json
import re
from typing import List, Optional, Tuple, Dict
from utils.bloom_rubric import BloomRubric

def rephrase_learning_outcome(learning_outcome: str, client) -> Optional[str]:
    """Rephrase the learning outcome to clarify its cognitive demands."""
    
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

def get_rephrase_respond_prediction_binary_class(learning_outcome: str, client) -> Dict[str, int]:
    """
    Get Rephrase and Respond binary prediction for all 6 Bloom categories.
    
    Args:
        learning_outcome: The learning outcome text
        client: OpenAI client
    
    Returns:
        Dictionary with 1/0 decisions for all 6 categories
    """
    categories = ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']
    
    rubric = BloomRubric()
    category_definitions = rubric.get_category_definitions()
    
    # Step 1: Rephrase the learning outcome
    rephrased = rephrase_learning_outcome(learning_outcome, client)
    if rephrased is None:
        rephrased = learning_outcome
    
    definitions_text = ""
    for cat in categories:
        definitions_text += f"{cat.upper()}: {category_definitions[cat]['description']}\n"

    # Step 2: Make binary decisions based on both versions
    prompt = f"""You are an expert educator classifying learning outcomes according to Bloom's Taxonomy.

BLOOM'S TAXONOMY CATEGORIES:
{definitions_text}

TASK: Analyze both versions of this learning outcome and decide which Bloom taxonomy categories apply. For each category, respond with 1 if it applies or 0 if it doesn't apply.

Original learning outcome: "{learning_outcome}"
Rephrased for clarity: "{rephrased}"

INSTRUCTIONS:
1. Consider both the original and rephrased versions
2. Identify the cognitive demands required
3. For each category, decide if it applies (1) or doesn't apply (0)
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
                {"role": "system", "content": "You are an expert at classifying learning outcomes using Bloom's Taxonomy. Consider both versions and provide JSON binary decisions."},
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
        print(f"Error in rephrase-respond binary classification: {str(e)}")
        return {cat: 0 for cat in categories}

def get_rephrase_respond_prediction_single_category(learning_outcome: str, client) -> str:
    """Get single best category using rephrase-respond approach."""
    binary_decisions = get_rephrase_respond_prediction_binary_class(learning_outcome, client)
    
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
def get_rephrase_respond_prediction_multi_class(learning_outcome: str, client) -> Dict[str, int]:
    """Alias for binary classification to maintain compatibility."""
    return get_rephrase_respond_prediction_binary_class(learning_outcome, client)
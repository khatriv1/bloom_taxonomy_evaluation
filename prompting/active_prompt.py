# bloom_taxonomy_evaluation/prompting/active_prompt.py
# FIXED: Added Self-Consistency to final predictions

import time
import json
import re
import numpy as np
from typing import List, Optional, Dict, Tuple
from collections import Counter
import pandas as pd
from utils.bloom_rubric import BloomRubric

class ActivePromptSelector:
    """Implements Active Prompting methodology with ROBUST column handling"""
    
    def __init__(self, pool_size: int = 20, k_samples: int = 2, consistency_samples: int = 5):
        self.pool_size = pool_size
        self.k_samples = k_samples
        self.consistency_samples = consistency_samples  # NEW: For self-consistency
        self.uncertainty_scores = {}
        
    def estimate_uncertainty(self, questions: List[str], client, category: str) -> Dict[str, float]:
        """Estimate uncertainty with MUCH BETTER prompts (unchanged)"""
        print(f"Estimating uncertainty for {len(questions)} questions in category: {category}")
        
        uncertainty_scores = {}
        
        for i, question in enumerate(questions):
            if (i + 1) % 5 == 0:
                print(f"Processing question {i + 1}/{len(questions)}")
            
            predictions = []
            for sample_idx in range(self.k_samples):
                pred = self._get_single_prediction(question, client, category)
                if pred is not None:
                    predictions.append(pred)
                time.sleep(0.05)
            
            if predictions:
                unique_predictions = len(set(predictions))
                disagreement = unique_predictions / len(predictions)
                uncertainty_scores[question] = disagreement
            else:
                uncertainty_scores[question] = 0.0
        
        return uncertainty_scores
    
    def _get_single_prediction(self, learning_outcome: str, client, category: str) -> Optional[int]:
        """Get a single binary prediction with MUCH BETTER prompts (unchanged)"""
        
        try:
            category_prompts = {
                'remember': {
                    'description': 'recalling facts, terms, basic concepts, or procedures',
                    'keywords': 'list, identify, name, recall, state, define, describe facts',
                    'example_yes': 'List the steps of mitosis',
                    'example_no': 'Analyze the effectiveness of mitosis in cell division'
                },
                'understand': {
                    'description': 'explaining ideas, interpreting, or summarizing concepts',
                    'keywords': 'explain, describe, interpret, summarize, discuss, outline',
                    'example_yes': 'Explain how photosynthesis works',
                    'example_no': 'List the steps of photosynthesis'
                },
                'apply': {
                    'description': 'using information in new situations or implementing procedures',
                    'keywords': 'use, apply, solve, demonstrate, implement, calculate',
                    'example_yes': 'Use the quadratic formula to solve equations',
                    'example_no': 'Explain what the quadratic formula is'
                },
                'analyze': {
                    'description': 'breaking down information, comparing, or examining relationships',
                    'keywords': 'analyze, compare, examine, break down, investigate, differentiate',
                    'example_yes': 'Compare different economic theories',
                    'example_no': 'Use economic theories to solve problems'
                },
                'evaluate': {
                    'description': 'making judgments, critiquing, or assessing value',
                    'keywords': 'evaluate, assess, judge, critique, justify, argue',
                    'example_yes': 'Evaluate the effectiveness of different teaching methods',
                    'example_no': 'Compare different teaching methods'
                },
                'create': {
                    'description': 'putting elements together to form something new or original',
                    'keywords': 'create, design, develop, construct, generate, produce',
                    'example_yes': 'Design a new experiment to test the hypothesis',
                    'example_no': 'Evaluate the effectiveness of the experiment'
                }
            }
            
            cat_info = category_prompts[category]
            
            prompt = f"""Does this learning outcome require {category.upper()} level thinking?

{category.upper()} means: {cat_info['description']}

Key indicators: {cat_info['keywords']}

Examples:
✅ YES - {category.upper()}: "{cat_info['example_yes']}"
❌ NO - NOT {category.upper()}: "{cat_info['example_no']}"

Learning Outcome: "{learning_outcome[:200]}"

Look for the action verbs and cognitive demands. Does this require students to {cat_info['description']}?

Answer "1" if YES (this requires {category} thinking)
Answer "0" if NO (this does NOT require {category} thinking)

Answer:"""

            response = client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": f"You are an expert in Bloom's Taxonomy. Carefully analyze if learning outcomes require {category} level thinking. Look at the action verbs and cognitive demands. Answer only 1 or 0."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=5,
                timeout=12
            )
            
            result = response.choices[0].message.content.strip()
            if result in ["1", "0"]:
                return int(result)
            
        except Exception as e:
            pass
            
        return None
    
    def select_uncertain_questions(self, uncertainty_scores: Dict[str, float], n_select: int = 2) -> List[str]:
        """Select the most uncertain questions"""
        sorted_questions = sorted(uncertainty_scores.items(), key=lambda x: x[1], reverse=True)
        selected = [q for q, score in sorted_questions[:n_select]]
        
        print(f"Selected {len(selected)} most uncertain questions:")
        for i, q in enumerate(selected):
            score = uncertainty_scores[q]
            print(f"  {i+1}. (uncertainty: {score:.3f}) {q[:60]}...")
        
        return selected

def find_learning_outcome_column(df: pd.DataFrame) -> str:
    """Robust column detection with detailed logging (unchanged)"""
    
    print(f"🔍 COLUMN DETECTION PROCESS:")
    print(f"Available columns: {list(df.columns)}")
    
    strategies = [
        ("Exact match", lambda df: _exact_match(df)),
        ("Partial match", lambda df: _partial_match(df)),
        ("Longest text", lambda df: _longest_text(df)),
        ("First text column", lambda df: _first_text_column(df))
    ]
    
    for strategy_name, strategy_func in strategies:
        try:
            result = strategy_func(df)
            if result:
                print(f"✅ {strategy_name} found: '{result}'")
                return result
            else:
                print(f"❌ {strategy_name} failed")
        except Exception as e:
            print(f"❌ {strategy_name} error: {e}")
    
    first_col = df.columns[0]
    print(f"⚠️ Using first column as fallback: '{first_col}'")
    return first_col

def _exact_match(df: pd.DataFrame) -> Optional[str]:
    """Strategy 1: Exact name matching"""
    possible_names = [
        'Learning_outcome', 'learning_outcome', 'outcome', 'Outcome',
        'text', 'Text', 'learning_objectives', 'objective',
        'description', 'Description', 'statement', 'Statement',
        'content', 'Content', 'learning_goal', 'goal'
    ]
    
    for col in df.columns:
        if col in possible_names:
            return col
    return None

def _partial_match(df: pd.DataFrame) -> Optional[str]:
    """Strategy 2: Partial name matching"""
    keywords = ['learning', 'outcome', 'objective', 'goal', 'description', 'statement']
    
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in keywords):
            return col
    return None

def _longest_text(df: pd.DataFrame) -> Optional[str]:
    """Strategy 3: Find column with longest text"""
    text_cols = []
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                avg_length = df[col].dropna().str.len().mean()
                if avg_length > 20:
                    text_cols.append((col, avg_length))
            except:
                continue
    
    if text_cols:
        return max(text_cols, key=lambda x: x[1])[0]
    return None

def _first_text_column(df: pd.DataFrame) -> Optional[str]:
    """Strategy 4: First text column"""
    for col in df.columns:
        if df[col].dtype == 'object':
            return col
    return None

def get_active_prompt_prediction_binary_class(learning_outcome: str, client, uncertainty_data: Dict = None,
                                            use_self_consistency: bool = True,
                                            consistency_samples: int = 5) -> Dict[str, int]:
    """Get active prompting binary prediction with SELF-CONSISTENCY"""
    categories = ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']
    results = {}
    
    for category in categories:
        # Get examples with CoT reasoning
        examples_text = ""
        if uncertainty_data and category in uncertainty_data:
            examples = uncertainty_data[category][:1]
            if examples:
                ex_outcome, ex_reasoning = examples[0]
                # Create detailed CoT reasoning
                detailed_reasoning = create_cot_reasoning_bloom(ex_outcome, category, ex_reasoning)
                examples_text = f"\nUncertain Example:\n"
                examples_text += f'Learning Outcome: "{ex_outcome[:80]}..."\n'
                examples_text += f'Reasoning: {detailed_reasoning}\n\n'
        
        category_prompts = {
            'remember': {
                'description': 'recalling facts, terms, basic concepts, or procedures',
                'keywords': 'list, identify, name, recall, state, define',
                'focus': 'retrieving factual information from memory'
            },
            'understand': {
                'description': 'explaining ideas, interpreting, or summarizing concepts',
                'keywords': 'explain, describe, interpret, summarize, discuss',
                'focus': 'demonstrating comprehension of meaning'
            },
            'apply': {
                'description': 'using information in new situations or implementing procedures',
                'keywords': 'use, apply, solve, demonstrate, implement, calculate',
                'focus': 'applying knowledge to solve problems'
            },
            'analyze': {
                'description': 'breaking down information, comparing, or examining relationships',
                'keywords': 'analyze, compare, examine, break down, investigate',
                'focus': 'examining parts and relationships'
            },
            'evaluate': {
                'description': 'making judgments, critiquing, or assessing value',
                'keywords': 'evaluate, assess, judge, critique, justify',
                'focus': 'making informed judgments based on criteria'
            },
            'create': {
                'description': 'putting elements together to form something new or original',
                'keywords': 'create, design, develop, construct, generate',
                'focus': 'combining elements into coherent wholes'
            }
        }
        
        cat_info = category_prompts[category]
        
        prompt = f"""Analyze this learning outcome for {category.upper()} level thinking in Bloom's Taxonomy.

{category.upper()}: {cat_info['description']}
Key indicators: {cat_info['keywords']}
Focus: Students will be {cat_info['focus']}

{examples_text}
LEARNING OUTCOME TO ANALYZE:
"{learning_outcome[:200]}"

STEP-BY-STEP ANALYSIS:
1. What are the action verbs? (look for: {cat_info['keywords']})
2. What cognitive level is required? 
3. Does this require {cat_info['focus']}?

Answer "1" if this requires {category.upper()} thinking
Answer "0" if this does NOT require {category.upper()} thinking

Answer:"""

        if not use_self_consistency:
            # Single prediction (original method)
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo-0125",
                    messages=[
                        {"role": "system", "content": f"You are an expert in Bloom's Taxonomy. Analyze learning outcomes carefully for {category} level thinking. Look at action verbs and cognitive demands. Answer only 1 or 0."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=5,
                    timeout=15
                )
                
                result = response.choices[0].message.content.strip()
                results[category] = 1 if result == "1" else 0
                        
            except Exception as e:
                print(f"Error in {category}: {e}")
                results[category] = 0
        
        else:
            # NEW: SELF-CONSISTENCY - Multiple predictions + most common answer
            predictions = []
            
            for i in range(consistency_samples):
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo-0125",
                        messages=[
                            {"role": "system", "content": f"You are an expert in Bloom's Taxonomy. Analyze learning outcomes carefully for {category} level thinking. Look at action verbs and cognitive demands. Answer only 1 or 0."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,  # Higher temperature for diversity
                        max_tokens=5,
                        timeout=15
                    )
                    
                    result = response.choices[0].message.content.strip()
                    if result == "1":
                        predictions.append(1)
                    elif result == "0":
                        predictions.append(0)
                    
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"Error in {category} sample {i+1}: {e}")
                    continue
            
            if predictions:
                # Take most common answer (SELF-CONSISTENCY)
                counter = Counter(predictions)
                most_common_answer = counter.most_common(1)[0][0]
                results[category] = most_common_answer
                print(f"Self-consistency for {category}: {predictions} → {most_common_answer}")
            else:
                results[category] = 0
        
        time.sleep(0.1)
    
    return results

def get_active_prompt_prediction_single_category(learning_outcome: str, client, uncertainty_data: Dict = None,
                                               use_self_consistency: bool = True,
                                               consistency_samples: int = 5) -> str:
    """Get single best category with SELF-CONSISTENCY"""
    binary_decisions = get_active_prompt_prediction_binary_class(learning_outcome, client, uncertainty_data,
                                                               use_self_consistency, consistency_samples)
    
    # IMPROVED CATEGORY SELECTION LOGIC
    category_priority = ['create', 'evaluate', 'analyze', 'apply', 'understand', 'remember']
    
    selected_categories = [cat for cat in category_priority if binary_decisions.get(cat, 0) == 1]
    
    if selected_categories:
        return selected_categories[0]
    else:
        return 'understand'

def prepare_active_prompting_data(df: pd.DataFrame, client, n_examples: int = 2) -> Dict[str, List[Tuple[str, str]]]:
    """Prepare active prompting examples with COMPLETE column fix (unchanged)"""
    print("Preparing Active Prompting data (COMPLETELY FIXED VERSION)...")
    
    categories = ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']
    active_examples = {}
    
    try:
        outcome_col = find_learning_outcome_column(df)
        print(f"✓ Successfully identified outcome column: '{outcome_col}'")
    except ValueError as e:
        print(f"❌ Column detection failed: {e}")
        return {cat: [] for cat in categories}
    
    max_samples = min(len(df), 10)
    sample_df = df.sample(n=max_samples, random_state=42)
    sample_questions = sample_df[outcome_col].tolist()
    
    selector = ActivePromptSelector(k_samples=2)
    
    for category in categories:
        print(f"\nProcessing category: {category}")
        
        try:
            uncertainty_scores = selector.estimate_uncertainty(sample_questions, client, category)
            
            if not uncertainty_scores:
                raise Exception("No uncertainty scores")
            
            selected_questions = selector.select_uncertain_questions(uncertainty_scores, n_examples)
            examples = create_active_examples(selected_questions, sample_df, category)
            active_examples[category] = examples
            
            print(f"✓ Created {len(examples)} examples for {category}")
            
        except Exception as e:
            print(f"⚠ Error in {category}: {e}")
            print(f"⚠ Using ground truth fallback...")
            
            try:
                examples = []
                possible_cat_cols = [category.capitalize(), category.lower(), category.upper(), category]
                found_col = None
                
                for cat_col in possible_cat_cols:
                    if cat_col in sample_df.columns:
                        found_col = cat_col
                        break
                
                if found_col:
                    positive_examples = sample_df[sample_df[found_col] == 1][outcome_col].tolist()
                    negative_examples = sample_df[sample_df[found_col] == 0][outcome_col].tolist()
                    
                    if positive_examples:
                        examples.append((positive_examples[0], f"This requires {category} thinking. Answer: 1"))
                    if negative_examples:
                        examples.append((negative_examples[0], f"This does not require {category} thinking. Answer: 0"))
                    
                    active_examples[category] = examples
                    print(f"✓ Created {len(examples)} fallback examples for {category}")
                else:
                    active_examples[category] = []
                    print(f"⚠ No column found for {category}")
            
            except Exception as fallback_error:
                print(f"⚠ Fallback failed for {category}: {fallback_error}")
                active_examples[category] = []
    
    return active_examples

def create_active_examples(selected_questions: List[str], ground_truth_data: pd.DataFrame, category: str) -> List[Tuple[str, str]]:
    """Create examples with BETTER column handling"""
    examples = []
    
    try:
        outcome_col = find_learning_outcome_column(ground_truth_data)
    except ValueError as e:
        print(f"❌ {e}")
        return []
    
    for question in selected_questions:
        matching_rows = ground_truth_data[ground_truth_data[outcome_col] == question]
        
        if not matching_rows.empty:
            possible_cat_cols = [category.capitalize(), category.lower(), category.upper(), category]
            label = 0
            
            for cat_col in possible_cat_cols:
                if cat_col in matching_rows.columns:
                    label = int(matching_rows.iloc[0][cat_col])
                    print(f"✓ Found category column '{cat_col}' for {category}")
                    break
            
            reasoning = create_detailed_reasoning(question, category, label)
            examples.append((question, reasoning))
    
    return examples

def create_detailed_reasoning(learning_outcome: str, category: str, label: int) -> str:
    """Create detailed reasoning for each example"""
    
    category_verbs = {
        'remember': ['list', 'identify', 'name', 'recall', 'state', 'define', 'describe', 'recognize'],
        'understand': ['explain', 'describe', 'interpret', 'summarize', 'discuss', 'outline', 'clarify'],
        'apply': ['use', 'apply', 'solve', 'demonstrate', 'implement', 'calculate', 'execute'],
        'analyze': ['analyze', 'compare', 'examine', 'break down', 'investigate', 'differentiate'],
        'evaluate': ['evaluate', 'assess', 'judge', 'critique', 'justify', 'argue', 'defend'],
        'create': ['create', 'design', 'develop', 'construct', 'generate', 'produce', 'compose']
    }
    
    outcome_lower = learning_outcome.lower()
    found_verbs = [verb for verb in category_verbs[category] if verb in outcome_lower]
    
    if label == 1:
        if found_verbs:
            reasoning = f"This requires {category} thinking because it uses the verb '{found_verbs[0]}' which requires students to {get_category_description(category)}. The cognitive demand clearly matches {category} level."
        else:
            reasoning = f"This requires {category} thinking as students need to {get_category_description(category)}. The overall cognitive demand is at the {category} level even without explicit keywords."
    else:
        actual_category = None
        for cat, verbs in category_verbs.items():
            if any(verb in outcome_lower for verb in verbs):
                actual_category = cat
                break
        
        if actual_category and actual_category != category:
            reasoning = f"This does NOT require {category} thinking. Instead, it requires {actual_category} level thinking based on the cognitive demands and action verbs used."
        else:
            reasoning = f"This does NOT require {category} thinking. The cognitive demands are at a different level of Bloom's taxonomy."
    
    return f"{reasoning} Answer: {label}"

def create_cot_reasoning_bloom(learning_outcome: str, category: str, basic_reasoning: str) -> str:
    """Create detailed Chain-of-Thought reasoning for Bloom taxonomy examples"""
    
    # Extract action verbs
    outcome_lower = learning_outcome.lower()
    category_verbs = {
        'remember': ['list', 'identify', 'name', 'recall', 'state', 'define'],
        'understand': ['explain', 'describe', 'interpret', 'summarize', 'discuss'],
        'apply': ['use', 'apply', 'solve', 'demonstrate', 'implement', 'calculate'],
        'analyze': ['analyze', 'compare', 'examine', 'break down', 'investigate'],
        'evaluate': ['evaluate', 'assess', 'judge', 'critique', 'justify'],
        'create': ['create', 'design', 'develop', 'construct', 'generate']
    }
    
    found_verbs = [verb for verb in category_verbs.get(category, []) if verb in outcome_lower]
    
    # Determine if it requires this category
    answer = "1" if "requires" in basic_reasoning.lower() else "0"
    
    if answer == "1":
        if found_verbs:
            detailed_reasoning = f"Let me analyze this step by step: 1) I identify the action verb '{found_verbs[0]}' which is a key indicator for {category} level thinking, 2) This verb requires students to {get_category_description(category)}, 3) The cognitive demand matches {category} because students must go beyond simple recall to engage in {category}-level processes, 4) The learning outcome structure aligns with {category} taxonomy level. Therefore, this requires {category} thinking. Answer: 1"
        else:
            detailed_reasoning = f"Let me analyze this step by step: 1) While specific {category} verbs aren't present, the overall cognitive demand requires {category} thinking, 2) Students must {get_category_description(category)} to achieve this outcome, 3) The complexity level and mental processes needed align with {category} taxonomy, 4) The learning expectations go beyond lower-order thinking. Therefore, this requires {category} thinking. Answer: 1"
    else:
        # Find what category it actually is
        actual_category = None
        for cat, verbs in category_verbs.items():
            if any(verb in outcome_lower for verb in verbs):
                actual_category = cat
                break
        
        if actual_category and actual_category != category:
            detailed_reasoning = f"Let me analyze this step by step: 1) I identify action verbs that suggest {actual_category} rather than {category}, 2) The cognitive demand requires students to {get_category_description(actual_category)}, 3) This is different from {category} which would require {get_category_description(category)}, 4) The learning outcome aligns with {actual_category} taxonomy level. Therefore, this does NOT require {category} thinking. Answer: 0"
        else:
            detailed_reasoning = f"Let me analyze this step by step: 1) The action verbs and cognitive demand don't align with {category} requirements, 2) Students are not required to {get_category_description(category)}, 3) The learning outcome represents a different taxonomy level, 4) The mental processes required are not characteristic of {category}. Therefore, this does NOT require {category} thinking. Answer: 0"
    
    return detailed_reasoning

def get_category_description(category: str) -> str:
    """Get description for each category"""
    descriptions = {
        'remember': 'recall and retrieve factual information',
        'understand': 'explain and interpret concepts',
        'apply': 'use knowledge in new situations',
        'analyze': 'break down and examine relationships',
        'evaluate': 'make judgments and assess value',
        'create': 'combine elements to form something new'
    }
    return descriptions.get(category, f'perform {category} level thinking')
# bloom_taxonomy_evaluation/utils/bloom_rubric.py

"""
Bloom Taxonomy Rubric Implementation - 6 Categories for Learning Outcome Classification
Based on Krathwohl's revised Bloom's Taxonomy
"""

class BloomRubric:
    """
    Bloom's Taxonomy 6-category rubric for classifying learning outcomes.
    Categories based on cognitive complexity from Remember to Create.
    """
    
    @staticmethod
    def get_category_definitions():
        """Return definitions for all 6 Bloom taxonomy categories."""
        return {
            "remember": {
                "description": "Retrieving relevant knowledge from long-term memory",
                "cognitive_processes": ["recognizing", "recalling"],
                "examples": [
                    "Students will list the steps of the scientific method",
                    "Students will identify the major organs of the human body",
                    "Students will recall the dates of World War II"
                ],
                "action_verbs": ["list", "identify", "recall", "name", "state", "define", "recognize", "memorize"]
            },
            
            "understand": {
                "description": "Determining the meaning of instructional messages, including oral, written, and graphic communication",
                "cognitive_processes": ["interpreting", "exemplifying", "classifying", "summarizing", "inferring", "comparing", "explaining"],
                "examples": [
                    "Students will explain the water cycle in their own words",
                    "Students will summarize the main themes of the novel",
                    "Students will describe the process of photosynthesis"
                ],
                "action_verbs": ["explain", "describe", "summarize", "interpret", "translate", "paraphrase", "discuss", "clarify"]
            },
            
            "apply": {
                "description": "Carrying out or using a procedure in a given situation",
                "cognitive_processes": ["executing", "implementing"],
                "examples": [
                    "Students will solve quadratic equations using the quadratic formula",
                    "Students will apply statistical methods to analyze data",
                    "Students will use proper citation format in their papers"
                ],
                "action_verbs": ["apply", "use", "implement", "execute", "solve", "demonstrate", "calculate", "operate"]
            },
            
            "analyze": {
                "description": "Breaking material into its constituent parts and detecting how the parts relate to one another and to an overall structure or purpose",
                "cognitive_processes": ["differentiating", "organizing", "attributing"],
                "examples": [
                    "Students will analyze the causes and effects of the Civil War",
                    "Students will compare and contrast different economic theories",
                    "Students will examine the relationship between variables in the dataset"
                ],
                "action_verbs": ["analyze", "compare", "contrast", "examine", "differentiate", "distinguish", "organize", "deconstruct"]
            },
            
            "evaluate": {
                "description": "Making judgments based on criteria and standards",
                "cognitive_processes": ["checking", "critiquing"],
                "examples": [
                    "Students will assess the validity of research methodology",
                    "Students will critique the effectiveness of different teaching strategies",
                    "Students will judge the credibility of news sources"
                ],
                "action_verbs": ["evaluate", "assess", "judge", "critique", "appraise", "defend", "justify", "argue"]
            },
            
            "create": {
                "description": "Putting elements together to form a novel, coherent whole or make an original product",
                "cognitive_processes": ["generating", "planning", "producing"],
                "examples": [
                    "Students will design an experiment to test their hypothesis",
                    "Students will develop an original business plan",
                    "Students will compose an original piece of music"
                ],
                "action_verbs": ["create", "design", "develop", "compose", "construct", "build", "generate", "invent"]
            }
        }
    
    @staticmethod
    def get_prompt_descriptions():
        """Return prompt-friendly descriptions for each category."""
        return {
            "remember": "The learning outcome requires students to retrieve, recall, or recognize facts, concepts, or procedures from memory without necessarily understanding or applying them.",
            
            "understand": "The learning outcome requires students to demonstrate comprehension by explaining, interpreting, summarizing, or describing concepts in their own words.",
            
            "apply": "The learning outcome requires students to use learned knowledge, skills, or procedures in specific situations or to solve problems using established methods.",
            
            "analyze": "The learning outcome requires students to break down information into parts, examine relationships between elements, or distinguish between different components.",
            
            "evaluate": "The learning outcome requires students to make judgments, assess quality, critique arguments, or determine value based on criteria or standards.",
            
            "create": "The learning outcome requires students to combine elements to form something new, original, or innovative, such as designing, planning, or producing novel solutions."
        }
    
    @staticmethod
    def get_category_hierarchy():
        """Return the hierarchical order of Bloom's taxonomy (simple to complex)."""
        return ["remember", "understand", "apply", "analyze", "evaluate", "create"]
    
    @staticmethod
    def get_action_verb_mapping():
        """Return mapping of common action verbs to Bloom categories."""
        definitions = BloomRubric.get_category_definitions()
        verb_mapping = {}
        
        for category, details in definitions.items():
            for verb in details["action_verbs"]:
                verb_mapping[verb.lower()] = category
        
        return verb_mapping
    
    @staticmethod
    def classify_by_action_verb(learning_outcome: str):
        """
        Simple classification based on action verbs in the learning outcome.
        Returns the most likely Bloom category based on verbs present.
        """
        verb_mapping = BloomRubric.get_action_verb_mapping()
        outcome_lower = learning_outcome.lower()
        
        # Count matches for each category
        category_scores = {"remember": 0, "understand": 0, "apply": 0, "analyze": 0, "evaluate": 0, "create": 0}
        
        for verb, category in verb_mapping.items():
            if verb in outcome_lower:
                category_scores[category] += 1
        
        # Return category with highest score
        max_category = max(category_scores, key=category_scores.get)
        max_score = category_scores[max_category]
        
        if max_score > 0:
            return max_category
        else:
            return "understand"  # Default fallback
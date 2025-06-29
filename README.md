# Bloom Taxonomy Binary Classification Evaluation

This project evaluates different prompting techniques for classifying learning outcomes using Bloom's Taxonomy with **binary classification**. It compares nine advanced LLM prompting strategies to determine which best matches human expert classifications of educational learning outcomes.

## Project Overview

The project uses **binary classification** to evaluate learning outcomes across all 6 Bloom taxonomy categories:
- **Remember**: Retrieving relevant knowledge from long-term memory
- **Understand**: Determining the meaning of instructional messages
- **Apply**: Carrying out or using a procedure in a given situation
- **Analyze**: Breaking material into constituent parts and detecting relationships
- **Evaluate**: Making judgments based on criteria and standards
- **Create**: Putting elements together to form a novel, coherent whole

### Binary Classification Approach
Unlike traditional single-label classification, this system:
- **Human experts** provide binary labels (1/0) for each of the 6 categories
- **AI models** predict binary decisions (1/0) for each category
- **Fair comparison** between human and AI binary predictions
- **Multi-label capability** - learning outcomes can belong to multiple categories

## Prompting Techniques Evaluated

1. **Zero-shot**: Direct binary classification using category definitions
2. **Chain of Thought (CoT)**: Step-by-step reasoning before binary classification
3. **Few-shot**: Provides binary examples before asking for classification
4. **Active Prompting**: Selects most informative examples for binary decisions
5. **Auto-CoT**: Automatically generates reasoning chains for binary classification
6. **Contrastive CoT**: Uses positive and negative reasoning for binary decisions
7. **Rephrase and Respond**: Rephrases outcome for clarity before binary classification
8. **Self-Consistency**: Multiple reasoning paths with majority voting for binary decisions
9. **Take a Step Back**: Derives principles before binary classification

## Directory Structure

```
bloom_taxonomy_evaluation/
├── data/
│   └── sample_full.csv        
├── prompting/
│   ├── zero_shot.py          
│   ├── cot.py                
│   ├── few_shot.py            
│   ├── active_prompt.py       
│   ├── auto_cot.py            
│   ├── contrastive_cot.py     
│   ├── rephrase_and_respond.py 
│   ├── self_consistency.py    
│   └── take_a_step_back.py    
├── utils/
│   ├── data_loader.py         
│   ├── bloom_rubric.py        
│   └── metrics.py     
├── evaluation/
│   ├── evaluate_zero_shot.py 
│   ├── evaluate_cot.py  
│   ├── evaluate_few_shot.py 
│   ├── evaluate_active_prompt.py 
│   ├── evaluate_auto_cot.py       
│   ├── evaluate_contrastive_cot.py  
│   ├── evaluate_rephrase_respond.py 
│   ├── evaluate_self_consistency.py 
│   └── evaluate_take_step_back.py   
├── results/                   
├── config.py                 
├── main.py                  
└── requirements.txt           
```

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/khatriv1/bloom_taxonomy_evaluation.git
cd bloom-taxonomy-evaluation
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Configure your settings in `config.py`:
```python
DATA_PATH = "data/sample_full.csv"
OPENAI_API_KEY = "your-openai-api-key-here"
```

5. Ensure the Bloom taxonomy dataset is in the data directory.

## Usage

### Run Complete Binary Evaluation
```bash
python main.py
```

You'll be prompted to:
1. Enter number of learning outcomes to evaluate (recommended: 50-100 for testing, blank for all 21,380)
2. Select which techniques to evaluate or run all

### Run Individual Technique
```bash
python evaluation/evaluate_zero_shot.py  # Test single technique
python evaluation/evaluate_few_shot.py   # Test few-shot approach
```

## Dataset Format

The Bloom taxonomy dataset contains 21,380 learning outcomes with expert scores for each Bloom category. The data loader:
- Handles both binary (0/1) and float (0.0-1.0) expert scores
- Converts scores to binary labels using threshold (default 0.5)
- Processes NaN/empty values as 0
- Validates data quality and reports statistics

Example data format:
```csv
Learning_outcome,Remember,Understand,Apply,Analyze,Evaluate,Create
"Students will list the major causes of WWI",1,0,0,0,0,0
"Students will analyze market trends",0,0,0,1,0,0
"Students will create a research proposal",0,0,0,0,0,1
```

## Evaluation Metrics

The project uses **5 key metrics**:

### Traditional Metrics:
1. **Primary Accuracy**: Percentage of times AI picks same main category as humans
2. **Cohen's Kappa (κ)**: Agreement beyond chance (-1 to 1, higher is better)
3. **Krippendorff's Alpha (α)**: Reliability measure (0 to 1, higher is better)
4. **Intraclass Correlation (ICC)**: Pattern correlation between human and AI

### Binary Classification Metric:
5. **Binary Accuracy**: Percentage of times AI matches humans on ALL 6 category decisions

### Per-Category Metrics:
- **Precision**: When AI says category applies, how often is it correct?
- **Recall**: Of all outcomes in category, how many did AI find?
- **F1 Score**: Harmonic mean of precision and recall

## Output

Results are saved in timestamped directories containing:

### Main Comparison Files:
- `bloom_all_techniques_comparison.csv` - All 5 metrics for all techniques
- `bloom_all_techniques_comparison.png` - Visual comparison charts
- `bloom_all_detailed_results.csv` - All predictions with binary accuracy
- `bloom_comprehensive_report.txt` - Detailed analysis and insights

### Per-Technique Files:
- `detailed_results.csv` - All predictions with primary and binary accuracy
- `binary_predictions.csv` - Raw 1/0 decisions for each category
- `metrics_summary.csv` - Summary of all 5 metrics
- `technique_performance.png` - 4-panel performance visualization

## Key Features

- **Binary Classification**: Each outcome can belong to multiple Bloom categories
- **Expert Ground Truth**: Uses expert binary labels for fair comparison
- **Data Isolation**: AI only sees learning outcome text, never human labels
- **Comprehensive Metrics**: 5 different ways to measure performance
- **Large Dataset**: 21,380 learning outcomes for robust evaluation
- **Flexible Execution**: Run all or selected techniques
- **Quality Validation**: Automatic data format detection and validation

## Requirements

- Python 3.7+
- OpenAI API key with GPT-3.5 access
- Required packages: numpy, pandas, matplotlib, seaborn, scikit-learn, scipy, krippendorff, openai

## Performance Expectations

### Good Performance Ranges:
- **Binary Accuracy**: 70-90%
- **Primary Accuracy**: 75-85%
- **Cohen's Kappa**: 0.6-0.8
- **F1 Scores**: 0.7-0.9 per category

### Dataset Recommendations:
- **Testing**: 50-100 outcomes for quick evaluation
- **Research**: 500+ outcomes for reliable statistics
- **Full Evaluation**: All 21,380 outcomes for comprehensive analysis

## Binary vs Traditional Classification

### Traditional (Single-Label):
```
Learning Outcome → Single Best Category
"Students will analyze data" → "analyze"
```

### Binary (Multi-Label):
```
Learning Outcome → Binary Decision for Each Category
"Students will analyze data" → {remember: 0, understand: 0, apply: 1, analyze: 1, evaluate: 0, create: 0}
```

## Notes

- **Data Format**: Automatically detects binary vs float data
- **Processing Time**: Depends on number of outcomes and techniques selected
- **API Rate Limits**: Handled automatically with delays
- **Ground Truth**: Determined by expert scores with proper threshold handling
- **Fair Evaluation**: Complete isolation between human labels and AI predictions

## Citation

If you are using our data or codes, please cite:

```
@inproceedings{2022.EDM-short-papers.55,
	title        = {Automatic Classification of Learning Objectives Based on Bloom's Taxonomy},
	author       = {Yuheng Li and Mladen Rakovic and Boon Xin Poh and Dragan Gasevic and Guanliang Chen},
	year         = 2022,
	month        = {July},
	booktitle    = {Proceedings of the 15th International Conference on Educational Data Mining},
	publisher    = {International Educational Data Mining Society},
	address      = {Durham, United Kingdom},
	pages        = {530--537},
	doi          = {10.5281/zenodo.6853191},
	isbn         = {978-1-7336736-3-1}
}
```

## Bloom's Taxonomy Reference

Based on Krathwohl's revised Bloom's Taxonomy:
- **Remember**: Recognition and recall of facts and basic concepts
- **Understand**: Comprehension and interpretation of meaning
- **Apply**: Use of knowledge in new situations
- **Analyze**: Breaking down information and examining relationships
- **Evaluate**: Making judgments based on criteria
- **Create**: Combining elements to form something new
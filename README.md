# Bloom Taxonomy Classification Evaluation

This project evaluates different prompting techniques for classifying learning outcomes using Bloom's Taxonomy. It compares multiple advanced LLM prompting strategies to determine which best matches human expert classifications of educational learning outcomes.

## Project Overview

The project classifies learning outcomes into 6 Bloom taxonomy categories:
- **Remember**: Retrieving relevant knowledge from long-term memory
- **Understand**: Determining the meaning of instructional messages
- **Apply**: Carrying out or using a procedure in a given situation
- **Analyze**: Breaking material into constituent parts and detecting relationships
- **Evaluate**: Making judgments based on criteria and standards
- **Create**: Putting elements together to form a novel, coherent whole

## Prompting Techniques Evaluated

1. **Zero-shot**: Direct classification using category definitions
2. **Chain of Thought (CoT)**: Step-by-step reasoning before classification
3. **Few-shot**: Provides examples before asking for classification
4. **Active Prompting**: Selects most informative examples using uncertainty sampling
5. **Auto-CoT**: Automatically generates reasoning chains
6. **Contrastive CoT**: Uses positive and negative reasoning
7. **Rephrase and Respond**: Rephrases outcome for clarity before classification
8. **Self-Consistency**: Multiple reasoning paths with majority voting
9. **Take a Step Back**: Derives principles before classification

## Directory Structure

```
bloom_taxonomy_evaluation/
├── data/
│   └── sample_full.csv         # Learning outcomes with Bloom classifications
├── prompting/
│   ├── zero_shot.py           # Zero-shot prompting
│   ├── cot.py                 # Chain of Thought
│   ├── few_shot.py            # Few-shot with examples
│   ├── active_prompt.py       # Active learning selection
│   ├── auto_cot.py            # Automatic CoT generation
│   ├── contrastive_cot.py     # Contrastive reasoning
│   ├── rephrase_and_respond.py # Clarification approach
│   ├── self_consistency.py    # Multiple sampling
│   └── take_a_step_back.py    # Abstract reasoning
├── utils/
│   ├── data_loader.py         # Loads and processes Bloom data
│   ├── bloom_rubric.py        # Category definitions and examples
│   └── metrics.py             # Evaluation metrics
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
├── results/                   # Generated results directory
├── config.py                  # Configuration and API keys
├── main.py                    # Main evaluation script
└── requirements.txt           # Python dependencies
```

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/your-repo/bloom-taxonomy-evaluation.git
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

4. Configure your OpenAI API key in `config.py`:
```python
OPENAI_API_KEY = "your-api-key-here"
```

5. Ensure the Bloom taxonomy dataset is in the data directory.

## Usage

### Run Complete Evaluation
```bash
python main.py
```

You'll be prompted to:
1. Enter number of learning outcomes to evaluate (recommended: 50-100 for testing)
2. Select which techniques to evaluate or run all

### Run Individual Technique
```bash
python evaluation/evaluate_zero_shot.py
```

## Dataset Format

The Bloom taxonomy dataset contains learning outcomes with expert scores for each Bloom category. The data loader:
- Processes expert scores for all 6 Bloom categories
- Determines ground truth using highest scoring category
- Prepares data for AI classification evaluation

Example data format:
```csv
Learning_outcome,Remember,Understand,Apply,Analyze,Evaluate,Create
"Students will list the major causes of WWI",0.8,0.2,0.1,0.3,0.1,0.0
"Students will analyze market trends",0.1,0.3,0.2,0.9,0.4,0.2
```

## Evaluation Metrics

The project uses 4 key metrics:

1. **Accuracy**: Percentage of exact matches between AI and human expert labels
2. **Cohen's Kappa (κ)**: Agreement beyond chance (-1 to 1, higher is better)
3. **Krippendorff's Alpha (α)**: Reliability measure (0 to 1)
4. **Intraclass Correlation (ICC)**: Pattern correlation between human and AI

## Output

Results are saved in timestamped directories containing:
- `bloom_all_techniques_comparison.csv` - Overall metrics comparison
- `bloom_all_techniques_comparison.png` - Visual comparison chart
- `bloom_all_detailed_results.csv` - All predictions
- `bloom_comprehensive_report.txt` - Detailed analysis
- Individual technique results in subdirectories

## Key Features

- **Single-label Classification**: Each outcome belongs to one primary Bloom category
- **Expert Ground Truth**: Uses expert scores to determine correct classifications
- **Fair Evaluation**: AI only sees learning outcome text, not expert scores
- **Comprehensive Metrics**: Multiple ways to measure performance
- **Flexible Execution**: Run all or selected techniques

## Requirements

- Python 3.7+
- OpenAI API key with GPT-3.5 access
- Required packages: numpy, pandas, matplotlib, seaborn, scikit-learn, scipy, krippendorff, openai

## Notes

- Minimum 10 learning outcomes recommended for meaningful results
- Processing time depends on number of outcomes and techniques selected
- API rate limits are handled automatically
- Ground truth is determined by highest expert score for each outcome

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
	isbn         = {978-1-7336736-3-1},
	abstract     = {Learning objectives, especially those well defined by applying Bloomâ€™s taxonomy for Cognitive Objectives, have been widely recognized as important in various teaching and learning practices. However, many educators have difficulties developing learning objectives appropriate to the levels in Bloomâ€™s taxonomy, as they need to consider the progression of learnersâ€™ skills with learning content as well as dependencies between different learning objectives. To remedy this challenge, we aimed to apply state-of-the-art computational techniques to automate the classification of learning objectives based on Bloomâ€™s taxonomy. Specifically, we collected 21,380 learning objectives from 5,558 different courses at an Australian university and manually labeled them according to the six cognitive levels of Bloomâ€™s taxonomy. Based on the labeled dataset, we applied five conventional machine learning approaches (i.e., naive Bayes, logistic regression, support vector machine, random forest, and XGBoost) and one deep learning approach based on pre-trained language model BERT to construct classifiers to automatically determine a learning objectiveâ€™s cognitive levels. In particular, we adopted and compared two methods in constructing the classifiers, i.e., constructing multiple binary classifiers (one for each cognitive level in Bloomâ€™s taxonomy) and constructing only one multi-class multi-label classifier to simultaneously identify all the corresponding cognitive levels. Through extensive evaluations, we demonstrated that: (i) BERT-based classifiers outperformed the others in all cognitive levels (Cohenâ€™s Kappa up to 0.93 and F1 score up to 0.95); (ii) three machine learning models â€" support vector machine, random forest, and XGBoost â€" delivered performance comparable to the BERT-based classifiers; and (iii) most of the binary BERT-based classifiers (5 out of 6) slightly outperformed the multi-class multi-label BERT-based classifier, suggesting that separating the characterization of different cognitive levels seemed to be a better choice than building only one model to identify all cognitive levels at one time.},
	editor       = {Antonija Mitrovic and Nigel Bosch}
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
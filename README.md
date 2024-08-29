# Sentiment Analysis and Aspect Extraction in Online Education

## 1. Sentiment Analysis Model Development

### 1.1 Dataset Preparation
- Input File: `UdemyComments.csv` and `Course_info.csv`
- File: `Dataset preparation.ipynb`
- Process: Downsample to 3 million comments, filter for English language, and consolidate course categories.
- Output File: `Udemy_grouped_by_category.csv`

### 1.2 Neutral Categorization Process
- Files: 
  - `Neutral classification.ipynb`: Separates test and train sets for different neutral definitions.
    - Input File: `Udemy_grouped_by_category.csv`
    - Output File: `train_strict.csv
test_strict.csv
train_moderate.csv
test_moderate.csv
train_bit_negative.csv 
test_bit_negative.csv
train_equal.csv
test_equal.csv'
  - `distillon4neutralsets.ipynb`: Evaluates DistilBERT and DistilRoBERTa on each neutral definition.
- Purpose: Determine the optimal definition of neutral sentiment for the Udemy rating system.

### 1.3 Dataset Size Optimization
- Files: 
  - `distillbert-dataset-size-explora...`
  - `distillroberta-dataset-size-expl...`
- Purpose: Evaluate and determine the optimal dataset size for model training, balancing performance and computational efficiency.

### 1.4 Hyperparameter Tuning
- Files: Various `.py`, `.txt`, and `.ipynb` files for BERT, RoBERTa, ELECTRA, ALBERT, and XLNet models.
- Process: Comprehensive hyperparameter optimization for five state-of-the-art transformer models using Bayesian optimization.
- Evaluation: Based on F1 score and overall running time.

### 1.5 Fine-tuning on Full Dataset
- Files: 
  - `Retrain_Electra_On_Full_Dataset.i...`
  - `Retrain_RoBERTa_On_Full_Dat...`
  - `Retrain_XLnet_On_Full_Datase...`
- Purpose: Retrain the top-performing models (ELECTRA, RoBERTa, XLNet) on the complete dataset using optimal hyperparameters.

### 1.6 Ensemble Approach
- File: `Ensemble_Approach_on_Three...`
- Process: Implement ensemble methods (Voting, Averaging, Confidence Weighted) to combine predictions from multiple models for improved accuracy.

## 2. Aspect Extraction Model Development

### 2.1 Keyword Labelling
- File: `Keyword labelling process.py`
- Purpose: Initial labelling of aspects based on predefined keywords.

### 2.2 Manual Data Selection
- File: `Gold_label_dataset_aspect_and...`
- Process: Manually curated dataset to ensure high-quality labelling for aspect classification.

### 2.3 Hyperparameter Tuning
- Files:
  - `(Code) Hyperparameter-Tuning.py`
  - `(Results) Hyperparameter-Tunin...`
- Purpose: Optimize hyperparameters for the aspect classification model.

### 2.4 First Round Fine-Tuning
- Files:
  - `(Code) 1st Round Fine-Tuning.py`
  - `(Results) 1st Round Fine-Tuning....`
- Process: Initial fine-tuning of the RoBERTa model for aspect classification.

### 2.5 Pseudo-Labelling
- Files:
  - `(Code) Pseudo- labelling.py`
  - `(Results) Pseudo-labelling.txt`
- Purpose: Expand the training dataset using high-confidence predictions from the initial model.

### 2.6 Second Round Fine-Tuning
- Files:
  - `(Code) 2nd Round Fine-tuning o...`
  - `(Results) 2nd Round Fine-tuning ...`
- Process: Final fine-tuning of the model using both manually labelled and pseudo-labelled data.

## Causal Inference Analysis

This section outlines the process of applying causal inference techniques to understand the factors affecting student satisfaction.

### 1. Aspect Extraction
- **Files**:
  - `Giving-aspect-1 million-to end.i...`
  - `Giving_aspect_1-500000.ipynb`
  - `Giving_aspect_500001-1 millio...`
- **Purpose**: Apply the developed aspect classification model to the full dataset

### 2. Sentiment Assignment
- **Files**: (Files in the "Giving Sentiment" folder)
- **Purpose**: Assign sentiment scores to the extracted aspects, removing ambiguous cases

### 3. Causal Analysis
- **Files**:
  - `Causal-Forest.ipynb`
  - `Exact-Matching.ipynb`
  - `Generalised-Addictive-Models.ip...`
- **Purpose**: Apply various causal inference techniques to identify factors influencing student satisfaction
- **Methods**:
  - Causal Forest
  - Exact Matching
  - Generalized Additive Models

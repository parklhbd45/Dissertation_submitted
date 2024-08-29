# Sentiment Analysis and Aspect Extraction in Online Education

# Sentiment Analysis and Aspect Extraction in Online Education

## 1. Sentiment Analysis Model Development

### 1.1 Dataset Preparation
- Input Files: 
  - `UdemyComments.csv`
  - `Course_info.csv`
- Process File: `Dataset preparation.ipynb`
- Methodology: 
  1. Downsampling to 3 million comments
  2. English language filtration
  3. Course category consolidation
- Output File: `Udemy_grouped_by_category.csv`

### 1.2 Neutral Categorization Process
#### 1.2.1 Test and Train Set Separation
- Process File: `Neutral classification.ipynb`
- Input File: `Udemy_grouped_by_category.csv`
- Output Files:
  - `train_strict.csv`, `test_strict.csv`
  - `train_moderate.csv`, `test_moderate.csv`
  - `train_bit_negative.csv`, `test_bit_negative.csv`
  - `train_equal.csv`, `test_equal.csv`

#### 1.2.2 Model Evaluation on Neutral Definitions
- Process File: `distillon4neutralsets.ipynb`
- Input Files: All output files from Section 1.2.1
- Models Evaluated: DistilBERT and DistilRoBERTa
- Output: Performance metrics (no file output)

Objective: Determine the optimal neutral sentiment definition for the Udemy rating system.

### 1.3 Dataset Size Optimization
- Process Files:
  - `distillbert-dataset-size-exploration.ipynb`
  - `distillroberta-dataset-size-exploration.ipynb`
- Input Files: 
  - `train_equal.csv`
  - `test_equal.csv`
- Methodology: Iterative model training on varying dataset sizes
- Evaluation Criteria: Performance metrics and computational efficiency
- Output: Optimal dataset size recommendation (no file output)

### 1.4 Hyperparameter Tuning
- Models: BERT, RoBERTa, ELECTRA, ALBERT, XLNet
- Process Files: Various `.py`, `.txt`, and `.ipynb` files for each model
- Input Files:
  - `train_equal.csv`
  - `test_equal.csv`
- Methodology: Bayesian optimization for hyperparameter search
- Evaluation Criteria: F1 score and computational efficiency
- Output: Optimal hyperparameter configurations for each model (no file output)

### 1.5 Full Dataset Fine-tuning
- Process Files:
  - `Retrain_Electra_On_Full_Dataset.ipynb`
  - `Retrain_RoBERTa_On_Full_Dataset.ipynb`
  - `Retrain_XLnet_On_Full_Dataset.ipynb`
- Input File: `test_equal.csv`
- Methodology: Model retraining using optimal hyperparameters from Section 1.4
- Output Models:
  1. `best_electra_model`
  2. `best_roberta_model`
  3. `best_xlnet_model`

### 1.6 Ensemble Approach
- Process File: `Ensemble_Approach_on_Three_Models.ipynb`
- Input Files:
  - `test_equal.csv`
  - Models from Section 1.5: `best_electra_model`, `best_roberta_model`, `best_xlnet_model`
- Methodology:
  1. Voting
  2. Averaging
  3. Confidence Weighted prediction
- Output: Final ensemble model performance metrics

## 2. Aspect Extraction Model Development

### 2.1 Keyword Labelling
- Process File: `Keyword labelling process.py`
- Input File: `Udemy_grouped_by_category.csv`
- Methodology: Automated aspect labelling based on predefined keywords
- Output File: `Udemy_comments_with_sentences_and_aspects.csv`

### 2.2 Manual Data Curation
- Input File: `Udemy_comments_with_sentences_and_aspects.csv`
- Methodology: Expert review and manual labelling for high-quality dataset creation
- Output File: `Gold_label_dataset_aspect_and_unlabel_recheck.csv`

### 2.3 Hyperparameter Optimization
- Process Files:
  - `(Code) Hyperparameter-Tuning.py`
  - `(Results) Hyperparameter-Tuning.txt`
- Input File: `Gold_label_dataset_aspect_and_unlabel_recheck.csv`
- Methodology: Grid search or Bayesian optimization for hyperparameter tuning
- Output: Optimal hyperparameter configuration (no file output)

### 2.4 Initial Model Fine-Tuning
- Process Files:
  - `(Code) 1st Round Fine-Tuning.py`
  - `(Results) 1st Round Fine-Tuning.txt`
- Input File: `Gold_label_dataset_aspect_and_unlabel_recheck.csv`
- Methodology: RoBERTa model fine-tuning using optimal hyperparameters from Section 2.3
- Output Model: `best_roberta_model_aspect_classification`

### 2.5 Pseudo-Labelling
- Process Files:
  - `(Code) Pseudo-labelling.py`
  - `(Results) Pseudo-labelling.txt`
- Input Files:
  - `Gold_label_dataset_aspect_and_unlabel_recheck.csv`
  - `best_roberta_model_aspect_classification` from Section 2.4
- Methodology: High-confidence prediction labelling on unlabeled data
- Output Files:
  - `combined_data_conf0.98_max300.csv`
  - `combined_data_conf0.98_max400.csv`
  - `combined_data_conf0.98_max500.csv`
  - `combined_data_conf0.99_max300.csv`
  - `combined_data_conf0.99_max400.csv`
  - `combined_data_conf0.99_max500.csv`

### 2.6 Final Model Fine-Tuning
- Process Files:
  - `(Code) 2nd Round Fine-tuning.py`
  - `(Results) 2nd Round Fine-tuning.txt`
- Input Files:
  - All output files from Section 2.5
  - `best_roberta_model_aspect_classification` from Section 2.4
- Methodology: RoBERTa model fine-tuning on combined manually labeled and pseudo-labeled data
- Output Model: `final_roberta_model_aspect_classification`

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

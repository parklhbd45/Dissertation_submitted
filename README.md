# Sentiment Analysis and Aspect Extraction in Online Education

## Abstract

This project aims to enhance the understanding of student satisfaction in online courses through advanced natural language processing and causal inference techniques. We focus on analyzing student feedback from the Udemy platform, employing state-of-the-art transformer models for sentiment analysis and aspect classification, followed by causal inference methods to identify key factors influencing overall course satisfaction.

Our sentiment analysis model, an ensemble of RoBERTa, ELECTRA, and XLNet, achieves an F1 score of 78.45%. The aspect classification model, based on RoBERTa with pseudo-labeling, attains an accuracy of 92.73%. These models are then leveraged in a causal inference framework to provide actionable insights for improving online course quality and student experience.

## Setup and Dependencies

To run this project, you need to install the following Python libraries: `pip install pandas numpy torch transformers scikit-learn datasets matplotlib seaborn tqdm optuna scipy econml pygam`

### File Compatibility
All `.ipynb` and `.py` files are interchangeable and can be run in both Python environments and Jupyter notebooks. Ensure that input and output file paths are correctly set in the code before execution.

## Note on File Types and Results Viewing

This project uses both Jupyter Notebooks (.ipynb) and Python scripts (.py). Here's how to view results for each:

### Jupyter Notebooks (.ipynb)
- Results are embedded within the notebook file itself.
- You can view the output cells directly in the notebook after running each cell.
- This allows for an interactive experience where code, explanations, and results are all in one place.

### Python Scripts (.py)
- Code and results are typically separated.
- The script will usually save results to separate output text files 
- You'll need to open these output files separately to view the results.
- Check the code or documentation for information on where output files are saved.

For example:
- When running `distillon4neutralsets.ipynb`, you'll see the performance metrics directly in the notebook.
- When running `(Code) Hyperparameter-Tuning.py`, the results will be saved in `(Results) Hyperparameter-Tuning.txt`.


# Dataset Source and Preparation

The dataset used in this project is sourced from Kaggle: [Udemy Courses Dataset](https://www.kaggle.com/datasets/hossaingh/udemy-courses)

Before running the scripts, please ensure you rename the following file:
- Rename `Comments.csv` to `UdemyComments.csv`

This renaming is necessary as the initial scripts expect the file to be named `UdemyComments.csv`.

---

## 1. Sentiment Analysis Model Development

### 1.1 Dataset Preparation
- **Input Files**: 
  - `UdemyComments.csv`
  - `Course_info.csv`
- **Process File**: `Dataset preparation.ipynb`
- **Methodology**: 
  1. Downsampling to 3 million comments
  2. English language filtration
  3. Course category consolidation
- **Output File**: `Udemy_grouped_by_category.csv`

### 1.2 Neutral Categorization Process
#### 1.2.1 Test and Train Set Separation
- **Process File**: `Neutral classification.ipynb`
- **Input File**: `Udemy_grouped_by_category.csv`
- **Output Files**:
  - `train_strict.csv`, `test_strict.csv`
  - `train_moderate.csv`, `test_moderate.csv`
  - `train_bit_negative.csv`, `test_bit_negative.csv`
  - `train_equal.csv`, `test_equal.csv`

#### 1.2.2 Model Evaluation on Neutral Definitions
- **Process File**: `distillon4neutralsets.ipynb`
- **Input Files**: All output files from Section 1.2.1
- **Models Evaluated**: DistilBERT and DistilRoBERTa
- **Output**: Performance metrics (no file output)

### 1.3 Dataset Size Optimization
- **Process Files**:
  - `distillbert-dataset-size-exploration.ipynb`
  - `distillroberta-dataset-size-exploration.ipynb`
- **Input Files**: 
  - `train_equal.csv`
  - `test_equal.csv`
- **Methodology**: Iterative model training on varying dataset sizes
- **Evaluation Criteria**: Performance metrics and computational efficiency
- **Output**: Optimal dataset size recommendation (no file output)

### 1.4 Hyperparameter Tuning
- **Models**: BERT, RoBERTa, ELECTRA, ALBERT, XLNet
- **Process Files**: Various `.py`, `.txt`, and `.ipynb` files for each model
- **Input Files**:
  - `train_equal.csv`
  - `test_equal.csv`
- **Methodology**: Bayesian optimization for hyperparameter search
- **Evaluation Criteria**: F1 score and computational efficiency
- **Output**: Optimal hyperparameter configurations for each model (no file output)

### 1.5 Full Dataset Fine-tuning
- **Process Files**:
  - `Retrain_Electra_On_Full_Dataset.ipynb`
  - `Retrain_RoBERTa_On_Full_Dataset.ipynb`
  - `Retrain_XLnet_On_Full_Dataset.ipynb`
- **Input File**: `test_equal.csv`
- **Methodology**: Model retraining using optimal hyperparameters from Section 1.4
- **Output Models**:
  1. `best_electra_model`
  2. `best_roberta_model`
  3. `best_xlnet_model`

### 1.6 Ensemble Approach
- **Process File**: `Ensemble_Approach_on_Three_Models.ipynb`
- **Input Files**:
  - `test_equal.csv`
  - Models from Section 1.5: `best_electra_model`, `best_roberta_model`, `best_xlnet_model`
- **Methodology**:
  1. Voting
  2. Averaging
  3. Confidence Weighted prediction
- **Output**: Final ensemble model with an accuracy of 78.45%

## 2. Aspect Extraction Model Development

### 2.1 Keyword Labelling
- **Process File**: `Keyword labelling process.py`
- **Input File**: `Udemy_grouped_by_category.csv`
- **Methodology**: Automated aspect labelling based on predefined keywords
- **Output File**: `Udemy_comments_with_sentences_and_aspects.csv`

### 2.2 Manual Data Curation
- **Input File**: `Udemy_comments_with_sentences_and_aspects.csv`
- **Methodology**: Expert review and manual labelling for high-quality dataset creation
- **Output File**: `Gold_label_dataset_aspect_and_unlabel_recheck.csv`

### 2.3 Hyperparameter Optimization
- **Process Files**:
  - `(Code) Hyperparameter-Tuning.py`
  - `(Results) Hyperparameter-Tuning.txt`
- **Input File**: `Gold_label_dataset_aspect_and_unlabel_recheck.csv`
- **Methodology**: Grid search or Bayesian optimization for hyperparameter tuning
- **Output**: Optimal hyperparameter configuration (no file output)

### 2.4 Initial Model Fine-Tuning
- **Process Files**:
  - `(Code) 1st Round Fine-Tuning.py`
  - `(Results) 1st Round Fine-Tuning.txt`
- **Input File**: `Gold_label_dataset_aspect_and_unlabel_recheck.csv`
- **Methodology**: RoBERTa model fine-tuning using optimal hyperparameters from Section 2.3
- **Output Model**: `best_roberta_model_aspect_classification`
Note
You can directly combine the code from Section 2.3 (Hyperparameter Optimization) and Section 2.4 (Initial Model Fine-Tuning) by running the hyperparameter tuning first, storing the optimal parameters, and then using those parameters to fine-tune the RoBERTa model in one script.

### 2.5 Pseudo-Labelling
- **Process Files**:
  - `(Code) Pseudo-labelling.py`
  - `(Results) Pseudo-labelling.txt`
- **Input Files**:
  - `Gold_label_dataset_aspect_and_unlabel_recheck.csv`
  - `best_roberta_model_aspect_classification` from Section 2.4
- **Methodology**: High-confidence prediction labelling on unlabeled data
- **Output Files**:
  - `combined_data_conf0.98_max300.csv`
  - `combined_data_conf0.98_max400.csv`
  - `combined_data_conf0.98_max500.csv`
  - `combined_data_conf0.99_max300.csv`
  - `combined_data_conf0.99_max400.csv`
  - `combined_data_conf0.99_max500.csv`

### 2.6 Final Model Fine-Tuning
- **Process Files**:
  - `(Code) 2nd Round Fine-tuning.py`
  - `(Results) 2nd Round Fine-tuning.txt`
- **Input Files**:
  - All output files from Section 2.5
  - `best_roberta_model_aspect_classification` from Section 2.4
- **Methodology**: RoBERTa model fine-tuning on combined manually labeled and pseudo-labeled data
- **Output Model**: `final_roberta_model_aspect_classification` with an accuracy of 92.73%

## 3. Causal Inference Analysis

### 3.1 Aspect and Sentiment Extraction
#### 3.1.1 Aspect Extraction
- **Process Files**: 
  - Giving-aspect-1 million-to end.ipynb`
  - Giving_aspect_1-500000.ipynb`
  - Giving_aspect_500001-1 million.ipynb`
- **Input**: Dataset with preprocessed comments
- **Methodology**: Apply the aspect classification model from Section 2.6
- **Output**: Dataset with extracted aspects for each comment
#### 3.1.2 Sentiment Assignment
- **Process Files**:
  - Giving-sentiment-first-chunk.ipynb`
  - Giving-sentiment-second-chunk.ipynb`
  - Giving-sentiment-the-last-chunk.ipynb`
- **Input**: Dataset with extracted aspects
- **Methodology**: Apply the ensemble sentiment analysis model from Section 1.6
- **Output**: Dataset with aspects and corresponding sentiment scores

### 3.2 Causal Analysis Methods
#### 3.2.1 Exact Matching
- **Process File**: Exact-Matching.ipynb`
- **Methodology**: 
  1. Match courses based on covariates except for the aspect of interest
  2. Calculate Average Treatment Effect (ATE) for each aspect
  3. Perform statistical significance testing using one-sample t-tests

#### 3.2.2 Generalized Additive Models (GAMs)
- **Process File**: Generalised-Addictive-Models.ipynb`
- **Methodology**:
  1. Model non-linear relationships between aspects and overall rating
  2. Include interaction terms to capture complex dependencies
  3. Use partial dependence plots to estimate ATEs

#### 3.2.3 Causal Forest with Double Machine Learning
- **Process File**: Causal-Forest.ipynb`
- **Methodology**:
  1. Implement causal forest algorithm for high-dimensional data and non-linear relationships
  2. Apply double machine learning to reduce bias in effect estimates
  3. Conduct sensitivity analyses with different hyperparameter settings

### 3.3 Comparative Analysis
- **Objective**: Synthesize results from all three causal inference methods
- **Methodology**:
  1. Identify consistent patterns and robust causal relationships across methods
  2. Analyze differences in aspect importance across course categories
  3. Investigate interaction effects between different aspects

### 3.4 Results Interpretation
- **Outputs**:
  1. Identification of most influential aspects affecting student satisfaction
  2. Category-specific insights for course improvement
  3. Discussion of implications for online education platforms and instructors




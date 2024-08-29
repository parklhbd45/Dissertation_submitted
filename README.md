# Dissertation_submitted
## Sentiment Analysis Model Development

This section details the process of developing and optimizing our sentiment analysis model for online course reviews.

### 1. Dataset Preparation
- **File**: `Dataset preparation.ipynb`
- **Purpose**: Initial data processing to create a manageable and relevant dataset.
- **Key steps**:
  - Downsample to 3 million records
  - Filter for English language comments
  - Consolidate course categories

### 2. Neutral Sentiment Definition Exploration
- **Files**: 
  - `Neutral classification.ipynb`: Separates test and train sets for different neutral definitions
  - `distillon4neutralsets.ipynb`: Evaluates DistilBERT and DistilRoBERTa on various neutral definitions
- **Objective**: Determine the optimal definition of neutral sentiment for our specific use case

### 3. Dataset Size Optimization
- **Files**:
  - `distillbert-dataset-size-explora...`
  - `distillroberta-dataset-size-expl...`
- **Purpose**: Identify the optimal dataset size balancing model performance and computational efficiency
- **Approach**: Evaluate model performance across different dataset sizes

### 4. Hyperparameter Tuning
- **Files**: Various `.py`, `.txt`, and `.ipynb` files for each model (BERT, RoBERTa, ELECTRA, ALBERT, XLNet)
- **Objective**: Optimize hyperparameters for each model to maximize performance
- **Method**: Utilize Bayesian optimization with Optuna library

### 5. Full Dataset Fine-tuning
- **Files**:
  - `Retrain_Electra_On_Full_Dataset.i...`
  - `Retrain_RoBERTa_On_Full_Dat...`
  - `Retrain_XLnet_On_Full_Datase...`
- **Purpose**: Retrain top-performing models on the full dataset with optimized hyperparameters

### 6. Ensemble Model Development
- **File**: `Ensemble_Approach_on_Three...`
- **Objective**: Combine top-performing models to create a more robust and accurate ensemble
- **Techniques**: Voting, Averaging, and Confidence-Weighted predictions

### Results
Our final sentiment analysis model achieves an F1 score of 78.45% using an Averaging Ensemble of RoBERTa, ELECTRA, and XLNet models.

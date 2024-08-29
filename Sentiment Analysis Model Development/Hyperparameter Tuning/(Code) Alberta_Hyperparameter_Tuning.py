
# ALBERT documentation: https://huggingface.co/albert/albert-base-v2\n",
# For hyperparameter optimization using Optuna with transformer models, refer to https://medium.com/carbon-consulting/transformer-models-hyperparameter-optimization-with-the-optuna-299e185044a8
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import AlbertTokenizer, AlbertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import re

# I fixed seed as If someone want to rerun my code, the result should be the same or very close
torch.manual_seed(42)
np.random.seed(42)

#Deleting URL and turn text into lowercase
def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = text.lower()
    return text

# I will balance dataset based on the minority class
def balance_data(df, samples_per_class = None):
    if samples_per_class is None:
        samples_per_class = df['sentiment_equal'].value_counts().min()
    balance_data = []
    for sentiment in df['sentiment_equal'].unique():
        class_data = df[df['sentiment_equal'] ==sentiment]
        balance_data.append(class_data.sample(samples_per_class, random_state=42))
    
    balance_df = pd.concat(balance_data, ignore_index=True)
    return balance_df

# Computing Metric
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
def tokenize_function(data: Dict[str, List], tokenizer: AlbertTokenizer, max_length):
    return tokenizer(data['comment'], padding='max_length', truncation=True, max_length=max_length)

def objective(trial: optuna.Trial, dataset: Dataset, trial_number):
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64])
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True)
    max_length = trial.suggest_categorical('max_length', [64, 128])
    
    print(f"\nTrial {trial_number}")
    print(f"Hyperparameters:")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {batch_size}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Max length: {max_length}")
    
    n_splits = 2
    test_size = 0.2
    fold_scores = []
    
    ss = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)
    
    for fold, (train_index, val_index) in enumerate(ss.split(dataset), 1):
        train_fold = dataset.select(train_index)
        val_fold = dataset.select(val_index)
        
        print(f"Fold {fold}:")
        print(f"  Train size: {len(train_fold)}")
        print(f"  Val size: {len(val_fold)}")
        
        
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        tokenized_train = train_fold.map(lambda x: tokenize_function(x, tokenizer, max_length), batched=True)
        tokenized_val = val_fold.map(lambda x: tokenize_function(x, tokenizer, max_length), batched=True)
        
        model = AlbertForSequenceClassification.from_pretrained('albert-base-v2', num_labels=3)
        
        training_args = TrainingArguments(
            output_dir=f'./results_trial_{trial_number}_fold_{fold}',
            num_train_epochs=2,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            evaluation_strategy="epoch",
            save_strategy="no",
            load_best_model_at_end=False,
            metric_for_best_model="f1",
            report_to="none"
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            compute_metrics=compute_metrics,
        )
        
        trainer.train()
        eval_result = trainer.evaluate()
        fold_scores.append(eval_result['eval_f1'])
        print(f"  Fold {fold} F1 score: {eval_result['eval_f1']:.4f}")
    mean_score = np.mean(fold_scores)
    print(f"Mean F1 score across folds: {mean_score:.4f}")
    
    return mean_score
train_df = pd.read_csv('/kaggle/input/equal-neutral-submit-version/train_equal.csv')
test_df = pd.read_csv('/kaggle/input/equal-neutral-submit-version/test_equal.csv')

for df in [train_df, test_df]:
    df['comment'] = df['comment'].apply(clean_text)
sentiment_mapping = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
train_df['label'] = train_df['sentiment_equal'].map(sentiment_mapping)
test_df['label'] = test_df['sentiment_equal'].map(sentiment_mapping)
balance_train_df = balance_data(train_df, samples_per_class=18750)
print(balance_train_df.head())  
print(balance_train_df['sentiment_equal'].value_counts())
balance_test_df = balance_data(test_df)
print(balance_test_df['sentiment_equal'].value_counts())
train_dataset = Dataset.from_pandas(balance_train_df)
test_dataset = Dataset.from_pandas(balance_test_df)
study = optuna.create_study(direction='maximize')
study.optimize(lambda trial: objective(trial, train_dataset, trial.number), n_trials=15)

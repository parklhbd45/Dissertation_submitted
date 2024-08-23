# Reference 1: Method for using the Trainer API in Hugging Face Transformers
# Source: https://www.analyticsvidhya.com/blog/2023/04/training-an-adapter-for-roberta-model-for-sequence-classification-task/

# Reference 2: Example of using the Trainer API with Roberta on Kaggle
# Source: https://www.kaggle.com/code/justinsiow/goodreads-roberta-with-trainer-api-torch

# Reference 3: Hyperparameter optimization using Optuna
# Source: https://medium.com/carbon-consulting/transformer-models-hyperparameter-optimization-with-the-optuna-299e185044a8


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import nltk
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import json

warnings.filterwarnings("ignore")

np.random.seed(42)
torch.manual_seed(42)


best_model_dir = './Best_Roberta_Model'
os.makedirs(best_model_dir, exist_ok=True)


df = pd.read_csv('Gold_label_dataset_aspect_and_unlabel_recheck.csv')
df = df.rename(columns={'Sentence': 'sentence', 'Label': 'label'})
labeled_data = df[df['label'].notna()]

print("Total samples:", len(df))
print("Labeled samples:", len(labeled_data))
print("Unlabeled samples:", len(df) - len(labeled_data))
print("\nInitial label distribution:")
print(labeled_data['label'].value_counts())

label_encoder = LabelEncoder()
labeled_data['label'] = label_encoder.fit_transform(labeled_data['label'])
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("\nLabel mapping:", label_mapping)

train_data, test_data = train_test_split(labeled_data, test_size=0.2, random_state=42, stratify=labeled_data['label'])

print("\nTrain set size:", len(train_data))
print("Test set size:", len(test_data))

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def tokenize_function(examples, max_length):
    return tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=max_length)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=1)
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def objective(trial, dataset, trial_number):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'per_device_train_batch_size': trial.suggest_categorical('per_device_train_batch_size', [32, 64]),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True),
        'max_length': trial.suggest_categorical('max_length', [64, 128])
    }
    
    print(f"\nTrial {trial_number}")
    print(f"Hyperparameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    n_splits = 2
    test_size = 0.2
    ss = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)
    best_f1_scores = []
    
    for fold, (train_index, val_index) in enumerate(ss.split(dataset), 1):
        print(f"\nFold {fold}:")
        train_fold = dataset.select(train_index)
        val_fold = dataset.select(val_index)
        
        print(f"  Train size: {len(train_fold)}")
        print(f"  Val size: {len(val_fold)}")
        
        tokenized_train = train_fold.map(lambda x: tokenize_function(x, params['max_length']), batched=True)
        tokenized_val = val_fold.map(lambda x: tokenize_function(x, params['max_length']), batched=True)
        
        model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=len(label_mapping))
        
        training_args = TrainingArguments(
            output_dir=f'./results_trial_{trial_number}_fold_{fold}',
            num_train_epochs=30,  # Reduced for faster hyperparameter tuning
            per_device_train_batch_size=params['per_device_train_batch_size'],
            per_device_eval_batch_size=params['per_device_train_batch_size'],
            weight_decay=params['weight_decay'],
            learning_rate=params['learning_rate'],
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
        best_f1_scores.append(eval_result['eval_f1'])
        print(f"  Fold {fold} F1 score: {eval_result['eval_f1']:.4f}")
    
    mean_f1_score = np.mean(best_f1_scores)
    print(f"Mean F1 score across folds: {mean_f1_score:.4f}")
    
    return mean_f1_score

print("\nStarting hyperparameter tuning...")
train_dataset = Dataset.from_pandas(train_data)
study = optuna.create_study(direction="maximize")
study.optimize(lambda trial: objective(trial, train_dataset, trial.number), n_trials=15)

print("\nBest trial:")
trial = study.best_trial
print("Value (Mean F1 score): ", trial.value)
print("Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

best_params = study.best_params

with open('best_hyperparameters.json', 'w') as f:
    json.dump(best_params, f)
print("Best hyperparameters saved to best_hyperparameters.json")

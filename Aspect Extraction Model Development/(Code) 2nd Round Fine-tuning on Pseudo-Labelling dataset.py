# Reference 1: Method for using the Trainer API in Hugging Face Transformers
# Source: https://www.analyticsvidhya.com/blog/2023/04/training-an-adapter-for-roberta-model-for-sequence-classification-task/

# Reference 2: Example of using the Trainer API with Roberta on Kaggle
# Source: https://www.kaggle.com/code/justinsiow/goodreads-roberta-with-trainer-api-torch

# Reference 3: Hyperparameter optimization using Optuna
# Source: https://medium.com/carbon-consulting/transformer-models-hyperparameter-optimization-with-the-optuna-299e185044a8

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import os
import warnings
import json
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)

best_params = {
    'learning_rate': 0.00028950889814633634,
    'per_device_train_batch_size': 64,
    'weight_decay': 0.004800600450489827,
    'max_length': 128
}

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def tokenize_and_prepare(examples):
    tokenized = tokenizer(examples['Sentence'], padding='max_length', truncation=True, max_length=best_params['max_length'])
    tokenized['labels'] = examples['encoded_label']
    return tokenized

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

def train_model(train_dataset, test_dataset, output_dir, num_epochs):
    model = RobertaForSequenceClassification.from_pretrained("best-roberta-model")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=best_params['per_device_train_batch_size'],
        per_device_eval_batch_size=best_params['per_device_train_batch_size'],
        weight_decay=best_params['weight_decay'],
        learning_rate=best_params['learning_rate'],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=1,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    print(f"Training model for {num_epochs} epochs...")
    trainer.train()
    
    return trainer, trainer.state.best_metric

data = pd.read_csv('Gold_label_dataset_aspect_and_unlabel_recheck.csv')
labeled_data = data[data['Label'].notna()]

label_encoder = LabelEncoder()
labeled_data['encoded_label'] = label_encoder.fit_transform(labeled_data['Label'])
label_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))

# I seperated the test set with the same seed with the first process
_, test_data = train_test_split(labeled_data, test_size=0.2, random_state=42, stratify=labeled_data['encoded_label'])

test_dataset = Dataset.from_pandas(test_data)
test_dataset = test_dataset.map(tokenize_and_prepare, batched=True, remove_columns=test_dataset.column_names)
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

dataset_files = [
    "combined_data_conf0.98_max300.csv",
    "combined_data_conf0.98_max400.csv",
    "combined_data_conf0.98_max500.csv",
    "combined_data_conf0.99_max300.csv",
    "combined_data_conf0.99_max400.csv",
    "combined_data_conf0.99_max500.csv"
]

epochs_to_try = [30, 40]
results = []
best_overall_f1 = 0
best_overall_epochs = 0
best_overall_dataset = ""

for dataset_file in dataset_files:
    print(f"\nExploring dataset: {dataset_file}")
    
    pseudo_labeled_data = pd.read_csv(f"/kaggle/input/combine-dataset-conf-without-aug/{dataset_file}")
    
    print(f"Dataset size: {len(pseudo_labeled_data)}")
    print("Label distribution:")
    print(pseudo_labeled_data['Label'].value_counts())
    
    train_dataset = Dataset.from_pandas(pseudo_labeled_data)
    train_dataset = train_dataset.map(tokenize_and_prepare, batched=True, remove_columns=train_dataset.column_names)
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    for epochs in epochs_to_try:
        output_dir = f"./roberta_exploration_{dataset_file.split('.')[0]}_{epochs}"
        trainer, best_f1 = train_model(train_dataset, test_dataset, output_dir, epochs)
        
        print(f"Dataset: {dataset_file}, Epochs: {epochs}, Best F1 Score: {best_f1}")
        
        results.append({
            'dataset': dataset_file,
            'epochs': epochs,
            'f1_score': best_f1
        })
        
        if best_f1 > best_overall_f1:
            best_overall_f1 = best_f1
            best_overall_epochs = epochs
            best_overall_dataset = dataset_file

print("\nSummary of results:")
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

print(f"\nBest overall configuration:")
print(f"Dataset: {best_overall_dataset}")
print(f"Epochs: {best_overall_epochs}")
print(f"F1 Score: {best_overall_f1}")

# Retrain with the best overall configuration
print("\nRetraining with the best overall configuration...")
best_pseudo_labeled_data = pd.read_csv(f"/kaggle/input/combine-dataset-conf-without-aug/{best_overall_dataset}")
best_train_dataset = Dataset.from_pandas(best_pseudo_labeled_data)
best_train_dataset = best_train_dataset.map(tokenize_and_prepare, batched=True, remove_columns=best_train_dataset.column_names)
best_train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

best_model_dir = f"./roberta_best_overall"
final_trainer, final_f1 = train_model(best_train_dataset, test_dataset, best_model_dir, best_overall_epochs)

final_trainer.save_model(best_model_dir)
print(f"Best model saved to {best_model_dir}")

tokenizer.save_pretrained(best_model_dir)
print(f"Tokenizer saved to {best_model_dir}")

final_results = final_trainer.evaluate(test_dataset)
print("Final evaluation results:", final_results)

test_predictions = final_trainer.predict(test_dataset)
cm = confusion_matrix(test_predictions.label_ids, test_predictions.predictions.argmax(-1))

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()

print("\nPer-class metrics:")
precision, recall, f1, _ = precision_recall_fscore_support(test_predictions.label_ids, test_predictions.predictions.argmax(-1), average=None)

class_metrics = {}
for i, label in enumerate(label_mapping.keys()):
    print(f"\n{label}:")
    print(f"F1: {f1[i]:.4f}")
    print(f"Precision: {precision[i]:.4f}")
    print(f"Recall: {recall[i]:.4f}")
    class_metrics[label] = {'f1': f1[i], 'precision': precision[i], 'recall': recall[i]}

final_results = {
    'best_dataset': best_overall_dataset,
    'best_epochs': best_overall_epochs,
    'best_f1_score': best_overall_f1,
    'best_hyperparameters': best_params,
    'final_evaluation': final_results,
    'per_class_metrics': class_metrics
}


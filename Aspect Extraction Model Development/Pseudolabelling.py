import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
import os
import warnings
import nltk
from nltk.corpus import wordnet
import random


data = pd.read_csv('Gold_label_dataset_aspect_and_unlabel_recheck.csv')
labeled_data = data[data['Label'].notna()]
unlabeled_data = data[data['Label'].isna()]

print(f"Total samples: {len(data)}")
print(f"Labeled samples: {len(labeled_data)}")
print(f"Unlabeled samples: {len(unlabeled_data)}")

label_encoder = LabelEncoder()
labeled_data['encoded_label'] = label_encoder.fit_transform(labeled_data['Label'])
label_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))

print("\nLabel mapping:")
for label, index in label_mapping.items():
    print(f"{label}: {index}")

print("\nLabel distribution in labeled data:")
print(labeled_data['Label'].value_counts())

train_data, test_data = train_test_split(labeled_data, test_size=0.2, random_state=42, stratify=labeled_data['encoded_label'])

print(f"\nTrain set size: {len(train_data)}")
print(f"Test set size: {len(test_data)}")

print("\nLabel distribution in train set:")
print(train_data['Label'].value_counts())

print("\nLoading model and tokenizer...")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("./Best_Roberta_Model")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def pseudo_label(df, confidence_threshold):
    pseudo_labeled = []
    valid_data = df[df['Sentence'].notna()]
    if valid_data.empty:
        print("No valid data found after filtering None values.")
        return pd.DataFrame()
    for _, row in tqdm(valid_data.iterrows(), total=len(valid_data), desc=f"Pseudo-labeling (conf={confidence_threshold})"):
        inputs = tokenizer(row['Sentence'], padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            probs = outputs.logits.softmax(dim=-1)
            confidence = probs.max().item()
            predicted_class = probs.argmax().item()
            if confidence > confidence_threshold:
                pseudo_labeled.append({
                    'Sentence': row['Sentence'],
                    'Label': label_encoder.inverse_transform([predicted_class])[0],
                    'encoded_label': predicted_class,
                    'confidence': confidence
                })
    return pd.DataFrame(pseudo_labeled)

none_count = unlabeled_data['Sentence'].isna().sum()
print(f"Number of None values: {none_count}")

results = {}

for confidence in [0.98, 0.99]:
    print(f"\nPerforming pseudo-labeling with confidence {confidence}")
    pseudo_labeled_data = pseudo_label(unlabeled_data.head(200000), confidence)
    print("\nPseudo-labeled data distribution:")
    print(pseudo_labeled_data['Label'].value_counts())
    results[confidence] = pseudo_labeled_data

for confidence, pseudo_data in results.items():
    for max_per_class in [300, 400, 500]:
        print(f"\nCreating dataset with confidence {confidence} and max_per_class {max_per_class}")
        selected_pseudo_data = pseudo_data.groupby('Label').head(max_per_class)
        combined_data = pd.concat([train_data, selected_pseudo_data], ignore_index=True)
        print("\nSelected pseudo-labeled data distribution:")
        print(selected_pseudo_data['Label'].value_counts())
        print("\nCombined data distribution:")
        print(combined_data['Label'].value_counts())
        filename = f"combined_data_conf{confidence}_max{max_per_class}.csv"
        output_dir = "./combined_datasets"
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, filename)
        combined_data.to_csv(file_path, index=False)
        print(f"\nDataset saved to {file_path}")
        print(f"Total samples: {len(combined_data)}")
        print(f"Original samples: {len(train_data)}")
        print(f"Pseudo-labeled samples: {len(selected_pseudo_data)}")

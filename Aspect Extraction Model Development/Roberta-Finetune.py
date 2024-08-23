
def train_model(train_dataset, test_dataset, output_dir, num_epochs):
    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=len(label_mapping))
    
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
        report_to="none"
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


train_dataset = train_dataset.map(lambda x: tokenize_function(x, best_params['max_length']), batched=True)
test_dataset = Dataset.from_pandas(test_data).map(lambda x: tokenize_function(x, best_params['max_length']), batched=True)

epochs_to_try = [10, 20, 30, 40, 50]
results = []

for epochs in epochs_to_try:
    print(f"\nTraining model for {epochs} epochs...")
    trainer, best_f1 = train_model(train_dataset, test_dataset, f"./model_epochs_{epochs}", epochs)
    results.append((epochs, best_f1))
    print(f"Best F1 score for {epochs} epochs: {best_f1}")


print("\nSummary of results:")
for epochs, f1_score in results:
    print(f"Epochs: {epochs}, Best F1 score: {f1_score}")


best_epochs, best_f1 = max(results, key=lambda x: x[1])
print(f"\nBest performance: {best_epochs} epochs with F1 score of {best_f1}")


print(f"\nRetraining model with best epochs: {best_epochs}")
final_trainer, final_f1 = train_model(train_dataset, test_dataset, best_model_dir, best_epochs)


final_trainer.save_model(best_model_dir)
print(f"Best model saved to {best_model_dir}")

tokenizer.save_pretrained(best_model_dir)
print(f"Tokenizer saved to {best_model_dir}")

print("\nEvaluating final model on test set...")
final_results = final_trainer.evaluate(test_dataset)
print("Final evaluation results:", final_results)

test_predictions = final_trainer.predict(test_dataset)

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
    'best_epochs': best_epochs,
    'best_f1_score': best_f1,
    'best_hyperparameters': best_params,
    'final_evaluation': final_results,
    'per_class_metrics': class_metrics
}

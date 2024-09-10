import os
import numpy as np
import json
import argparse
from transformers import (
    AutoConfig,
    BertForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from utils_data import load_data, MyDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Run BERT for sequence classification.")
    parser.add_argument('--data_root', type=str, default='data', help='Root directory for data.')
    parser.add_argument('--model', type=str, default='bert-base-uncased', help='Model identifier from Hugging Face models.')
    parser.add_argument('--output_dir', type=str, default='experiments', help='Output directory to save trained model.')
    parser.add_argument('--epoch', type=int, default=2, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate.')
    parser.add_argument('--bs', type=int, default=32, help='Batch size.')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length.')
    parser.add_argument('--seed', type=int, default=666, help='Random seed for reproducibility.')
    return parser.parse_args()

def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1) if isinstance(p.predictions, np.ndarray) else p.predictions
    correct = (preds == p.label_ids).sum()
    return {'accuracy': correct / len(preds)}

def main():
    args = parse_args()
    print("Input Arguments:", json.dumps(vars(args), indent=2, sort_keys=False))

    set_seed(args.seed)  # Set seed for reproducibility

    config = AutoConfig.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = BertForSequenceClassification.from_pretrained(args.model, config=config)

    train_data = load_data(os.path.join(args.data_root, 'train.json'))
    train_dataset = MyDataset(train_data, tokenizer, args.max_length)
    eval_data = load_data(os.path.join(args.data_root, 'val.json'))
    eval_dataset = MyDataset(eval_data, tokenizer, args.max_length)
    test_data = load_data(os.path.join(args.data_root, 'test.json'))
    test_dataset = MyDataset(test_data, tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epoch,
        learning_rate=args.lr,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        logging_dir=f"{args.output_dir}/logs",
        logging_strategy="steps",
        evaluation_strategy="steps",
        save_strategy="epoch",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    if training_args.do_train:
        train_result = trainer.train()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_model()
        trainer.save_state()

    if training_args.do_eval:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    if training_args.do_predict:
        predictions = trainer.predict(test_dataset).predictions
        predictions = np.argmax(predictions, axis=1)
        output_file = os.path.join(args.output_dir, "predict_results.txt")
        with open(output_file, "w") as writer:
            writer.write("index\tprediction\n")
            for index, prediction in enumerate(predictions):
                writer.write(f"{index}\t{prediction}\n")

if __name__ == "__main__":
    main()

import argparse
import logging
import os
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser(description="Prepare dataset for the first task")
parser.add_argument(
    "--config",
    type=str,
    default="task1/config.yaml",
    help="Config file",
)

def tokenize_batch(batch):
    prefixed_articles = ["summarize: " + article for article in batch["article"]]

    article_tokens = tokenizer(
        prefixed_articles,
        truncation=True,
        max_length=512,
        padding="max_length"
    )

    highlights_tokens = tokenizer(
        batch["highlights"],
        truncation=True,
        max_length=150,
        padding="max_length"
    )

    return {
        "input_ids": article_tokens["input_ids"],
        "attention_mask": article_tokens["attention_mask"],
        "labels": highlights_tokens["input_ids"] 
    }

if __name__ == '__main__':
    logger.info("Starting prepare data script")

    # Read config
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"Config found: {args.config}")

    # Set global params
    model_name = config['model']
    seed = config['seed']

    logger.info(f"Using model: {model_name}")
    logger.info(f"Using seed: {seed}")

    # Load the dataset
    raw_dataset = load_dataset(config['dataset']['name'], config['dataset']['data_dir'])

    logger.info(f"Loaded dataset")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logger.info(f"Loaded tokenizer")

    # Count of samples
    train_samples = config['dataset']['train_samples']
    test_samples = config['dataset']['test_samples']

    # Sometimes dataset is too huge and we need to take only part of it
    raw_dataset["train"] = raw_dataset["train"].shuffle(seed=seed).select(range(train_samples))
    raw_dataset["test"] = raw_dataset["test"].shuffle(seed=seed).select(range(test_samples))

    logger.info(f"Shuffled and selected {train_samples} samples for training and {test_samples} samples for testing")

    # Tokenize the dataset
    tokenized_dataset = raw_dataset.map(tokenize_batch, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["id", "article", "highlights"])
    tokenized_dataset.set_format("torch")

    logger.info(f"Tokenized dataset")

    # Save the dataset
    output_dir = os.path.join(config['dataset']['output_dir'], 'tokenized_dataset', os.path.basename(config['dataset']['name']))
    tokenized_dataset.save_to_disk(output_dir)
    logger.info(f"Saved tokenized dataset to {output_dir}")

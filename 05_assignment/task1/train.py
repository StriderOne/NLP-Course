import argparse
import logging
import torch
import yaml
from transformers import AutoTokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_from_disk

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser(description='Training script')
parser.add_argument(
    '--config',
    type=str,
    default='task1/config.yaml',
    help='Config file',
)

def convert_types(value):
    if isinstance(value, str):
        if value.lower() == 'true':
            return True
        if value.lower() == 'false':
            return False
        
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value
    return value

if __name__ == '__main__':
    logger.info('Starting prepare data script')

    # Read config
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f'Config found: {args.config}')

    # Load dataset
    dataset = load_from_disk('data/tokenized_dataset/cnn_dailymail')
    logger.info(f'Dataset loaded: data/tokenized_dataset/cnn_dailymail')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = T5ForConditionalGeneration.from_pretrained(config['model']).to(device)
    logger.info(f'Using model: {config["model"]}')
    logger.info(f'Using device: {config["seed"]}')

    correctly_typed_params = {
        key: convert_types(value) 
        for key, value in config['training']['trainer_params'].items()
    }

    training_args = TrainingArguments(**correctly_typed_params)

    trainer = Trainer(
        model,
        training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
    )

    trainer.train()
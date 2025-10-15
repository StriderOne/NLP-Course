import argparse
import logging
import torch
from peft import TaskType, LoraConfig, AdaLoraConfig, IA3Config
from pprint import pprint
import yaml
import os
from transformers import AutoTokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, BitsAndBytesConfig
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

SAFE_EVAL_CONTEXT = {
    "torch": torch,
    "TaskType": TaskType,
}

def convert_value(value: str):
    """
    Пытается преобразовать строковое значение в соответствующий тип данных.
    """
    if not isinstance(value, str):
        return value

    # Простое преобразование
    if value.lower() == 'true':
        return True
    if value.lower() == 'false':
        return False
    if value.lower() == 'none':
        return None

    # Попытка преобразовать в число (int, float)
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            pass # Не является числом, идем дальше

    # Попытка выполнить строку как код Python в безопасном окружении
    try:
        # Используем eval с ограниченным доступом к глобальным переменным
        return eval(value, SAFE_EVAL_CONTEXT)
    except (NameError, SyntaxError, AttributeError):
        # Если eval не сработал, значит это обычная строка (например, путь к файлу)
        return value

def process_config_values(data):
    """
    Рекурсивно обходит словарь или список и преобразует значения с помощью convert_value.
    """
    if isinstance(data, dict):
        # Если это словарь, обрабатываем каждое значение
        return {key: process_config_values(value) for key, value in data.items()}
    elif isinstance(data, list):
        # Если это список, обрабатываем каждый элемент
        return [process_config_values(item) for item in data]
    else:
        # Для всех остальных типов (уже преобразованных или строковых) вызываем конвертер
        return convert_value(data)

def create_adapter(config, **kwargs):
    name = config['name']
    params = config['params']
    adapter_config = None

    if name == 'lora':
        adapter_config = LoraConfig(**params)
    elif name == 'qlora':
        adapter_config = LoraConfig(**params)
    elif name == 'adalora':
        adapter_config = AdaLoraConfig(**params, **kwargs)
    elif name == 'IA3': 
        adapter_config = IA3Config(**params)
    else:
        raise ValueError(f'Unknown PEFT method: {name}')

    return adapter_config

if __name__ == '__main__':
    logger.info('Starting prepare data script')

    # Read config
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = process_config_values(config)
    logger.info(f'Config found: {args.config}')

    # Load dataset
    dataset_path = os.path.join(config['dataset']['output_dir'], 'tokenized_dataset', config['dataset']['name'].split('/')[-1])
    dataset_path = config['training'].get('dataset_path', dataset_path)
    dataset = load_from_disk(dataset_path)
    logger.info(f'Dataset loaded: {dataset_path}')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    logger.info(f'Using device: {config["seed"]}')

    training_args = TrainingArguments(**config['training']['trainer_params'])
    total_step = (len(dataset["train"]) // training_args.per_device_train_batch_size) * training_args.num_train_epochs

    for peft_method in config['training']['peft_methods']:
        if peft_method['name'] == 'qlora':
            bnb_config = BitsAndBytesConfig(**peft_method['bnb_config'])
            model = T5ForConditionalGeneration.from_pretrained(config['model'], quantization_config=bnb_config).to(device)
        else:
            model = T5ForConditionalGeneration.from_pretrained(config['model']).to(device)
        logger.info(f'Using model: {config["model"]}')
        adapter_config = create_adapter(peft_method, total_step=total_step)
        logger.info(f'Training with PEFT method: {peft_method["name"]}')
        model.add_adapter(adapter_config, adapter_name=peft_method['name'])

        trainer = Trainer(
            model,
            training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
        )
        logger.info(f'Training...')
        trainer.train()

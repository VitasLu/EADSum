import argparse

from datasets import DatasetDict
from transformers import AutoTokenizer

from data_utils import CNNDMDatasetLoader
from metrics import compute_equation_acc, compute_metrics_equation, compute_metrics_equation_aux
from train_utils import train_and_evaluate


def run(args):
    #### Prepare datasets
    if args.dataset == 'cnndm':
        dataset_loader = CNNDMDatasetLoader()
    else:
        raise ValueError

    datasets = dataset_loader.load_from_json()

    if args.llm is None:
        pass
    elif args.llm == 'gpt':
        train_llm_rationales = dataset_loader.load_llm_preds(split='train')
    else:
        raise ValueError

    if args.llm is not None:
        datasets['train'] = datasets['train'].add_column('llm_rationale', train_llm_rationales)

    if args.subsample < 1.0:
        datasets['train'] = datasets['train'].train_test_split(test_size=1.0-args.subsample, seed=args.run)['train']

    if dataset_loader.has_valid:
        if args.llm is None:
            pass
        elif args.llm == 'gpt':
            valid_llm_rationales, valid_llm_labels = dataset_loader.load_llm_preds(split='valid')
        else:
            raise ValueError

        datasets['valid'] = datasets['valid'].add_column('llm_label', valid_llm_labels)
        datasets['valid'] = datasets['valid'].add_column('llm_rationale', valid_llm_rationales)
    else:
        train_valid_datasets = datasets['train'].train_test_split(test_size=0.1, seed=0)

        datasets = DatasetDict({
            'train': train_valid_datasets['train'],
            'valid': train_valid_datasets['test'],
        })

    if args.label_type == 'gt':
        pass
    elif args.label_type == 'llm' and args.llm is not None: # Standard Fine-tuning
        train_label_acc = compute_equation_acc(datasets['train']['llm_label'], datasets['train']['label'])
        test_label_acc = compute_equation_acc(datasets['test']['llm_label'], datasets['test']['label'])

        print(f'LLM Train Acc: {train_label_acc:.4f}')
        print(f'LLM Test Acc: {test_label_acc:.4f}')

        datasets['train'] = datasets['train'].remove_columns('label')
        datasets['train'] = datasets['train'].add_column('label', datasets['train']['llm_label'])
    else:
        raise ValueError

    if args.llm is not None:
        if 'rationale' in datasets['train'].column_names:
            datasets = datasets.remove_columns('rationale')
 
        datasets = datasets.rename_column('llm_rationale', 'rationale')


    #### Prepare datasets Prepare data for training
    tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained)
    tokenizer.save_pretrained(f'./tokenizer')

    if args.model_type == 'task_prefix' and args.llm is not None:
        def tokenize_function(examples):
            model_inputs = tokenizer(['Summarize: ' + text for text in examples['input']], max_length=args.max_input_length, truncation=True)
            expl_model_inputs = tokenizer(['Rationale: ' + text for text in examples['input']], max_length=args.max_input_length, truncation=True)
            model_inputs['expl_input_ids'] = expl_model_inputs['input_ids']
            model_inputs['expl_attention_mask'] = expl_model_inputs['attention_mask']
            
            with tokenizer.as_target_tokenizer():
                label_output_encodings = tokenizer(['Summary: ' + text for text in examples['label']], max_length=256, truncation=True)
                rationale_output_encodings = tokenizer(['Rationale: ' + text for text in examples['rationale']], max_length=256, truncation=True)
            
            model_inputs['labels'] = label_output_encodings['input_ids']
            model_inputs['aux_labels'] = rationale_output_encodings['input_ids']
        
            return model_inputs

    elif args.model_type == 'standard':
        def tokenize_function(examples):
            model_inputs = tokenizer(
                examples['input'],
                max_length=args.max_input_length,
                truncation=True
            )

            with tokenizer.as_target_tokenizer():
                label_output_encodings = tokenizer(examples['label'], max_length=256, truncation=True)

            model_inputs['labels'] = label_output_encodings['input_ids']

            return model_inputs

    else:
        raise ValueError


    if args.llm is None:
        tokenized_datasets = datasets.map(
            tokenize_function,
            remove_columns=['input', 'label'],
            batched=True
        )
    else:
        tokenized_datasets = datasets.map(
            tokenize_function,
            remove_columns=['input', 'rationale', 'label'],
            batched=True
        )


    if args.model_type == 'standard':
        compute_metrics = compute_metrics_equation_aux(tokenizer)

    else:
        compute_metrics = compute_metrics_equation(tokenizer)


    train_and_evaluate(args, args.run, tokenizer, tokenized_datasets, compute_metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True ,default='cnndm')
    parser.add_argument('--subsample', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--max_steps', type=int, default=20000) # default 50000
    parser.add_argument('--eval_steps', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--optimizer_name', type=str, default='AdamW')
    parser.add_argument('--lr', type=float, default=5e-5) # default 5e-5
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--from_pretrained', type=str, default='google-t5/t5-base')
    # parser.add_argument('--from_pretrained', type=str, default='google/t5-v1_1-base')
    parser.add_argument('--label_type', type=str, default='gt')
    parser.add_argument('--llm', type=str, default='gpt') 
    parser.add_argument('--max_input_length', type=int, default=512)
    parser.add_argument('--grad_steps', type=int, default=8) # default 1
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--gen_max_len', type=int, default=128) # default 16
    parser.add_argument('--parallelize', action='store_true')
    parser.add_argument('--model_type', type=str, default='task_prefix')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--output_rationale', action='store_true')

    args = parser.parse_args()

    run(args)

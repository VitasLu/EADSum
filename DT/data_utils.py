import argparse
import re
import json
import numpy as np

from datasets import Dataset, DatasetDict, load_dataset


DATASET_ROOT = 'datasets'


class DatasetLoader(object):
    def __init__(self, dataset_name, source_dataset_name, dataset_version, has_valid, split_map,
                 batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=None):
        self.data_root = DATASET_ROOT
        self.dataset_name = dataset_name
        self.source_dataset_name = source_dataset_name
        self.dataset_version = dataset_version
        self.has_valid = has_valid
        self.split_map = split_map

        self.batch_size = batch_size
        self.train_batch_idxs = train_batch_idxs
        self.test_batch_idxs = test_batch_idxs
        self.valid_batch_idxs = valid_batch_idxs
        
        assert self.split_map is not None    


    def load_from_source(self):
        if self.source_dataset_name is None:
            self.source_dataset_name = self.dataset_name
        if self.dataset_version is None:
            datasets = load_dataset(self.source_dataset_name)
        else:
            datasets = load_dataset(self.source_dataset_name, self.dataset_version)
        return datasets


    def to_json(self, datasets):
        for k, v in self.split_map.items():
            datasets[v].to_json(f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_{k}.json')


    def load_from_json(self):
        data_files = {
            'train': f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_train.json',
            'test': f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_test.json',
        }

        if self.has_valid:
            data_files.update({'valid': f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_valid.json',})

        datasets = load_dataset('json', data_files=data_files)
        datasets = self._post_process(datasets) 

        # subsample training dataset if needed
        num_train = len(datasets['train'])
        idxs = list()
        for idx in self.train_batch_idxs:
            idxs += range(idx*self.batch_size, (idx+1)*self.batch_size)        
        datasets['train'] = Dataset.from_dict(datasets['train'][[idx for idx in idxs if idx < num_train]])

        return datasets


    def load_llm_preds(self, split):
        labels = list()
        rationales = list()

        for idx in getattr(self, f'{split}_batch_idxs'):
            with open(f'{self.data_root}/{self.dataset_name}/llm/{split}_CoT_{idx}.json') as f:
                outputs = json.load(f)

            for output in outputs:
                rationale, label = self._parse_llm_output(output)

                rationales.append(rationale)
                labels.append(label)

        return rationales, labels


    def load_gpt_preds(self, split):
        labels = list()
        rationales = list()
        
        with open(f'{self.data_root}/gpt-neox/{self.dataset_name}/{split}.json') as f:
            outputs = json.load(f)

        for output in outputs:
            rationale, label = self._parse_gpt_output(output)

            rationales.append(rationale)
            labels.append(label)

        return rationales, labels


    def _post_process(self, datasets):
        raise NotImplementedError


    def _parse_llm_output(self, output):
        raise NotImplementedError


    def _parse_gpt_output(self, output):
        raise NotImplementedError


class CNNDMDatasetLoader(DatasetLoader):
    def __init__(self):
        dataset_name = 'cnndm'
        source_dataset_name = 'cnndm'
        dataset_version = None
        has_valid = False
        split_map = {
            'train': 'train',
            'test': 'test',
        }
        batch_size = 78706
        train_batch_idxs = range(1)
        test_batch_idxs = range(1)


        super().__init__(dataset_name, source_dataset_name, dataset_version, has_valid, split_map,
                 batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=None)


    def load_from_source(self):
        # Load Training set from source
        with open(f'{self.data_root}/{self.dataset_name}/train/cnndm_train_70835.json') as f: 
            train_dataset = json.load(f)   
        
        # Load Test set from source
        with open(f'{self.data_root}/{self.dataset_name}/train/cnndm_test_7871.json') as f:
            test_dataset = json.load(f)   

        dataset1 = list()
        for data in train_dataset:
            input = f'{data["src"]}'
            equation = f'{data["original_summary"]}' # gpt3.5_cot_summary
            rationale = f'{data["element-aware"]}'

            dataset1.append({
                'input': input,
                'label': equation,
                'rationale': rationale,
            })

        ## Training set and test set are in the same file
        # idxs = np.random.RandomState(seed=0).permutation(len(dataset))
        # train_idxs = idxs[:180]
        # test_idxs = idxs[181:]
        # train_dataset = Dataset.from_list(np.array(dataset)[train_idxs].tolist())
        # test_dataset = Dataset.from_list(np.array(dataset)[test_idxs].tolist())

        # Training set and test set are in different files
        train_dataset = Dataset.from_list(np.array(dataset1).tolist())
        
        dataset2 = list()
        for data in test_dataset:
            input = f'{data["src"]}'
            equation = f'{data["original_summary"]}' # gpt3.5_cot_summary
            rationale = f'{data["element-aware"]}'

            dataset2.append({
                'input': input,
                'label': equation,
                'rationale': rationale,
            })

        test_dataset = Dataset.from_list(np.array(dataset2).tolist()) 

        datasets = DatasetDict({
            'train': train_dataset,
            'test': test_dataset
        })

        return datasets
        

    def _post_process(self, datasets):
        return datasets

    def load_llm_preds(self, split):
        labels = list()
        rationales = list()

        # Load Training set from source
        with open(f'{self.data_root}/{self.dataset_name}/train/cnndm_train_70835.json') as f:
            train_dataset = json.load(f)

        # Load Test set from source
        with open(f'{self.data_root}/{self.dataset_name}/train/cnndm_test_7871.json') as f:
            test_dataset = json.load(f)

        if split == 'train':
            for data in train_dataset:
                rationale = f'{data["element-aware"]}'
                label = f'{data["original_summary"]}'

                rationales.append(rationale)
                labels.append(label)

        else:
            for data in test_dataset:
                rationale = f'{data["element-aware"]}'
                label = f'{data["original_summary"]}'

                rationales.append(rationale)
                labels.append(label)

        return rationales, labels

    # def _parse_llm_output(self, output):
    #     rationale_label = output.split('Q:')[0]
    #     rationale_label = rationale_label.rstrip()
    #     try:
    #         rationale, label = output.split('The answer is')
    #     except:
    #         rationale = ' '
    #         label = ' '
    #         return rationale, label
            
    #     rationale = rationale.rstrip()
    #     try:
    #         label = re.search(r'\(.*\)', label).group(0)
    #     except:
    #         label = ' '

    #     return rationale, label

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()

    if args.dataset == 'cnndm':
        dataset_loader = CNNDMDatasetLoader()
    else:
        raise ValueError

    datasets = dataset_loader.load_from_source()
    dataset_loader.to_json(datasets)
    rationales, labels = dataset_loader.load_llm_preds(split='train')
    # print(rationales, labels)
from torch.utils.data import Dataset
import json
from transformers import AutoTokenizer
import torch


def pad_chinese_labels(labels, ignore_id=-100, max_length=100):
    new_labels = []
    pad_ids = [ignore_id] * max_length

    for i, label_ids in enumerate(labels):
        label_ids = (label_ids + pad_ids)[:max_length]
        new_labels.append(label_ids)

    new_labels = torch.tensor(new_labels)
    return new_labels


def tokenize_and_align_labels(labels, all_word_ids, max_length=60,
                              ignore_id=-100, label_all_tokens=True):
    new_labels = []
    pad_ids = [ignore_id] * max_length

    for i, label in enumerate(labels):
        word_ids = all_word_ids[i]
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        label_ids = (label_ids + pad_ids)[:max_length]
        new_labels.append(label_ids)

    new_labels = torch.tensor(new_labels)

    return new_labels


class TokenClassificationDataset(Dataset):
    def __init__(self,
                 df,
                 label_encoder,
                 is_chinese=False
                 ):

        if not is_chinese:
            self.max_length = 60
            model_name = './lm/bert-large-uncased'
        else:
            self.max_length = 100
            model_name = './lm/bert-large-chinese'

        x = df['sentence']
        self.length = len(x)

        y = df['label'] #num, seq_len
        self.n_classes = len(label_encoder.classes_)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        x_tokenized = self.tokenizer(x,
                                     padding='max_length',
                                     truncation=True,
                                     max_length=self.max_length,
                                     is_split_into_words=True,
                                     return_tensors='pt')

        self.input_ids = x_tokenized['input_ids']
        self.attention_mask = x_tokenized['attention_mask']

        if not is_chinese:
            all_word_ids = [x_tokenized.word_ids(batch_index=i)
                            for i in range(self.length)]

            y = label_encoder.transform(y)
            self.labels = tokenize_and_align_labels(y, all_word_ids)
        else:
            self.labels = label_encoder.transform(y)
            self.labels = pad_chinese_labels(self.labels)

    def __getitem__(self, index) -> dict:
        input_ids = self.input_ids[index]
        attention_mask = self.attention_mask[index]
        labels = self.labels[index]

        return {'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
                }

    def __len__(self):
        return self.length


def solve_conll2003_data(train_file, val_file, test_file, label_encoder):

    def solve(file):
        lines = open(file, 'r').readlines()[2:]

        dataset = {'sentence': [], 'label': []}
        tokens = []
        labels = []

        for line in lines:
            line = line.strip()
            if line == '':
                dataset['sentence'].append(tokens)
                dataset['label'].append(labels)
                tokens = []
                labels = []
            else:
                objs = line.split()
                token, label = objs[0], objs[-1]
                tokens.append(token)
                labels.append(label)

        return dataset

    train_df = solve(train_file)
    val_df = solve(val_file)
    test_df = solve(test_file)

    label_encoder.fit(['B-PER', 'I-PER', 'B-ORG', 'I-ORG',
                       'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC', 'O'])

    train_dataset = TokenClassificationDataset(train_df, label_encoder)
    val_dataset = TokenClassificationDataset(val_df, label_encoder)
    test_dataset = TokenClassificationDataset(test_df, label_encoder)
    return train_dataset, val_dataset, test_dataset, label_encoder


def solve_conll2004_data(train_file, val_file, test_file, label_encoder):

    def solve(file):
        data = json.load(open(file, 'r'))
        dataset = {'sentence': [], 'label': []}

        for item in data:
            tokens = item['tokens']
            entities = item['entities']

            labels = ['O'] * len(tokens)

            for ent in entities:
                type = ent['type']
                start = ent['start']
                end = ent['end']

                if type == 'Peop':
                    labels[start] = 'B-PER'
                    labels[start + 1 : end + 1] = ['I-PER'] * (end - start)
                elif type == 'Org':
                    labels[start] = 'B-ORG'
                    labels[start + 1 : end + 1] = ['I-ORG'] * (end - start)
                elif type == 'Loc':
                    labels[start] = 'B-LOC'
                    labels[start + 1 : end + 1] = ['I-LOC'] * (end - start)

            dataset['sentence'].append(tokens)
            dataset['label'].append(labels)

        return dataset

    train_df = solve(train_file)
    val_df = solve(val_file)
    test_df = solve(test_file)

    label_encoder.fit(['B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'O'])

    train_dataset = TokenClassificationDataset(train_df,
                                               label_encoder,

                                               )

    val_dataset = TokenClassificationDataset(val_df,
                                             label_encoder,

                                             )

    test_dataset = TokenClassificationDataset(test_df,
                                              label_encoder,

                                              )
    return train_dataset, val_dataset, test_dataset, label_encoder


def solve_msra_data(train_file, val_file, test_file, label_encoder):

    def solve(file):

        dataset = {'sentence': [], 'label': []}

        for line in open(file, 'r').readlines():
            tokens, labels = line.strip().split('\t')
            tokens = tokens.strip().split('\x02')
            labels = labels.strip().split('\x02')

            dataset['sentence'].append(tokens)
            dataset['label'].append(labels)

        return dataset

    train_df = solve(train_file)
    val_df = solve(val_file)
    test_df = solve(test_file)

    label_encoder.fit(['B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'O'])

    train_dataset = TokenClassificationDataset(train_df,
                                               label_encoder,
                                               is_chinese=True)

    val_dataset = TokenClassificationDataset(val_df,
                                             label_encoder,
                                             is_chinese=True)

    test_dataset = TokenClassificationDataset(test_df,
                                              label_encoder,
                                              is_chinese=True)

    return train_dataset, val_dataset, test_dataset, label_encoder
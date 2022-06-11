import json
from torch.utils.data import Dataset
import pandas as pd
import torch
import shutil, os


class DataFrameTextClassificationDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 label_encoder,
                 x_label='text',
                 y_label='label',
                 ):

        self.x = df[x_label]
        self.length = len(self.x)

        if y_label is not None:
            self.y = label_encoder.transform(df[y_label])
            self.y = pd.Series(self.y)
            self.n_classes = len(label_encoder.classes_)
        else:
            self.y = None

    def __getitem__(self, index) -> dict:
        x = self.x.iloc[index]
        if self.y is not None:
            y = self.y.iloc[index]
            return {'x': str(x), 'y': int(y)}
        else:
            return {'x': str(x)}

    def __len__(self):
        return self.length


def convert_json_file_to_df_data(file):
    lines = open(file, 'r', encoding='utf-8').readlines()

    head = json.loads(lines[0]).keys()

    data = {h: [] for h in head}

    for line in lines:
        item = json.loads(line)
        for k, v in item.items():
            data[k].append(v)

    return pd.DataFrame(data)


def write_result_to_file(ret, label_encoder, saved_file):
    predictions = []
    for _, logits in ret:
        a, y_hat = torch.max(logits, dim=1)
        predictions += y_hat.tolist()

    predictions = label_encoder.inverse_transform(predictions)
    fo = open(saved_file, 'w+')
    fo.write('index\tlabel\n')
    for i, label in enumerate(predictions):
        fo.write('{}\t{}\n'.format(i, label))
    print('\n' * 5)


def creat_file(file):
    if not os.path.exists(os.path.abspath(file)):
        file_abs_path = os.path.abspath(file)
        abs_dir, abs_filename = os.path.split(file_abs_path)
        if not os.path.exists(abs_dir):
            os.makedirs(abs_dir)
        with open(file_abs_path, mode="w", encoding="utf-8") as _:
            pass


def clear_dir(dir_path):
    shutil.rmtree(dir_path)
    os.mkdir(dir_path)


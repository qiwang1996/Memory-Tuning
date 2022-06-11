from seqeval.metrics import f1_score
import torch
import shutil, os


class LabelEncoder:
    def __init__(self):
        self.label2id = {}
        self.id2label = {}
        self.classes_ = []

    def fit(self, init_labels):
        for i, label in enumerate(init_labels):
            self.label2id[label] = i
            self.id2label[i] = label
        self.classes_ = init_labels

    def transform(self, y):
        new_y = [[] for _ in range(len(y))]

        for i, seq in enumerate(y):
            for label in seq:
                new_y[i].append(self.label2id[label])
        return new_y


def eval_func(y_predict, y_true, label_encoder: LabelEncoder, ignore_id=-100):
    Y_predict, Y_true = [], []

    y_predict = y_predict.tolist()
    y_true = y_true.tolist()

    for y_ps, y_ts in zip(y_predict, y_true):
        temp_p, temp_t = [], []
        for y_p, y_t in zip(y_ps, y_ts):
            if y_t != ignore_id:
                temp_p.append(label_encoder.id2label[y_p])
                temp_t.append(label_encoder.id2label[y_t])
        Y_true.append(temp_t)
        Y_predict.append(temp_p)

    return f1_score(Y_true, Y_predict), Y_true, Y_predict


def clear_dir(dir_path):
    shutil.rmtree(dir_path)
    os.mkdir(dir_path)

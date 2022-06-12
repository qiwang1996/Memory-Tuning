from typing import Optional

import fire
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from adapters.bert import add_bert_adapters, AdapterConfig, freeze_all_parameters, unfreeze_bert_adapters
from data import *
from models import AutoModelForTokenClassificationFinetuner

from utils import *

import shutil, os


os.environ["TOKENIZERS_PARALLELISM"] = "true"

def train(
    mode,
    task,
    train_file,
    val_file,
    test_file,
    saved_file,
    preseqlen=None,
    adapter_size=None,
    memory_num=None,
    model_name: str = './lm/bert-base-uncased',
    num_epochs: int = 30,
    n_workers: int = 4,
    gpus: int = 2,
    precision: int = 32,
    patience: int = 10,
    lr: float = 2e-05,
    batch_size: int = 128,
):
    print('------------*Start task:{}, mode:{}, memory_num:{}*------------'.format(task, mode, memory_num))
    torch.random.manual_seed(42)
    task_list = ['CoNLL2003', 'CoNLL2004', 'msra']
    solve_list = [solve_conll2003_data, solve_conll2004_data, solve_msra_data]

    assert task in task_list

    if task == 'msra':
        model_name = './lm/bert-base-chinese'

    solve_func = solve_list[task_list.index(task)]

    label_encoder = LabelEncoder()

    train_dataset, val_dataset, test_dataset, label_encoder = solve_func(train_file, val_file, test_file, label_encoder)

    test_loader = DataLoader(test_dataset, num_workers=n_workers,
                             batch_size=2318)

    # Load pre-trained model (weights)
    model = AutoModelForTokenClassificationFinetuner(eval_func,
                                                     label_encoder,
                                                     mode,
                                                     model_name,
                                                     n_classes=train_dataset.n_classes,
                                                     preseqlen=preseqlen,
                                                     lr=lr)

    if mode == 'adapter' or mode == 'prefix+adapter':
        # Add adapters and freeze all layers
        config = AdapterConfig(
            hidden_size=768, adapter_size=adapter_size,
            adapter_act='relu', adapter_initializer_range=1e-2,
            memory_num=memory_num, adapter_layer_norm_eps=1e-12,
        )
        model.model.bert = add_bert_adapters(model.model.bert, config)
        model.model.bert = freeze_all_parameters(model.model.bert)

        # Unfreeze adapters and the classifier head
        model.model.bert = unfreeze_bert_adapters(model.model.bert)
        model.model.classifier.requires_grad = True
    elif mode == 'prefix':
        model.model.bert = freeze_all_parameters(model.model.bert)

    dir = os.path.join(os.path.abspath(os.getcwd()), 'saved_model', mode)
    best_para = torch.load(os.path.join(dir, os.listdir(dir)[0]))['state_dict']

    model.load_state_dict(best_para)

    trainer = pl.Trainer(max_epochs=0,
                         gpus=1,
                         precision=precision)
    
    with torch.no_grad():
        ret = trainer.predict(model, test_loader, return_predictions=True)
    predictions = []
    labels = []

    for _, logits in ret:
        _, y_pre = torch.max(logits, dim=2)
        predictions += y_pre.tolist()

    for batch in test_loader:
        labels += batch['labels'].cpu().tolist()

    predict_file = './data/msra/pred.txt'
    label_file = './data/msra/label.txt'

    fp = open(predict_file, 'w+')
    fl = open(label_file, 'w+')
    for p, l in zip(predictions, labels):
        p_tmp, l_tmp = [], []
        for i in range(len(l)):
            if str(l[i]) != '-100':
                p_tmp.append(label_encoder.id2label[int(p[i])])
                l_tmp.append(label_encoder.id2label[int(l[i])])

        fp.write(' '.join(p_tmp) + '\n')
        fl.write(' '.join(l_tmp) + '\n')

if __name__ == '__main__':
    fire.Fire(train)

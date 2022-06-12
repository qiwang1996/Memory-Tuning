from typing import Optional

import fire
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from adapters import add_bert_adapters, AdapterConfig, freeze_all_parameters, unfreeze_bert_adapters
from data import *
from models import AutoModelForSequenceClassificationFinetuner

from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score
from scipy.stats import spearmanr

from sklearn import preprocessing

import shutil, os

from transformers.adapters import MAMConfig, PrefixTuningConfig

import transformers.adapters as ta




import shutil, os

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
    num_epochs: int = 50,
    n_workers: int = 4,
    gpus: int = 2,
    precision: int = 32,
    patience: int = 10,
    lr: float = 2e-05,
    batch_size: int = 64, 
):
    print('------------*Start task:{}, mode:{}, memory_num:{}*------------'.format(task, mode, memory_num))
    torch.random.manual_seed(42)
    task_list = ['CoLA', 'MNLI', 'MSRP', 'QNLI', 'QQP', 'RTE', 'SST-2', 'STS-B']
    solve_list = [solve_cola_data, solve_mnli_data, solve_msrp_data,
                  solve_qnli_data, solve_qqp_data, solve_rte_data,
                  solve_sst_data, solve_sts_data]
    metric_list = [matthews_corrcoef, accuracy_score, f1_score, accuracy_score,
                   f1_score, accuracy_score, accuracy_score, spearmanr]

    assert task in task_list
    solve_func = solve_list[task_list.index(task)]
    eval_func = metric_list[task_list.index(task)]

    label_encoder = preprocessing.LabelEncoder()

    train_dataset, val_dataset, test_dataset, label_encoder = solve_func(train_file, val_file, test_file, label_encoder)

    train_loader = DataLoader(train_dataset, num_workers=n_workers,  
                              shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, num_workers=n_workers,
                            batch_size=batch_size, persistent_workers=True)
    test_loader = DataLoader(test_dataset, num_workers=n_workers,
                             batch_size=872)

    # Load pre-trained model (weights)
    model = AutoModelForSequenceClassificationFinetuner(eval_func, mode,
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
    elif mode == 'mam_adapter':
        prefix_cfg = PrefixTuningConfig(flat=False, prefix_length=16,
                                        non_linearity='relu', dropout=0.1)
        adapter_cfg = ta.AdapterConfig(
                                    reduction_factor=48,
                                    non_linearity='relu',
                                    mh_adapter=False,
                                    output_adapter=True,
                                    scaling=4.0,
                                    is_parallel=True)
        cfg = MAMConfig(prefix_cfg, adapter_cfg)
        model.model.bert.add_adapter("mam_adapter", config=cfg)
        model.model.bert.train_adapter('mam_adapter')
        model.model.bert.set_active_adapters("mam_adapter")
    elif mode == 'bitfit':
        model.model.bert = freeze_all_parameters(model.model.bert)
        for name, sub_module in model.named_modules():
            for param_name, param in sub_module.named_parameters():
                if 'bias' in param_name:
                    param.requires_grad = True

    dir = os.path.join(os.path.abspath(os.getcwd()), 'saved_model', mode)
    
    best_para = torch.load(os.path.join(dir, os.listdir(dir)[0]))['state_dict']
    model.load_state_dict(best_para)

    trainer = pl.Trainer(max_epochs=0,
                         gpus=1,
                         precision=precision)

    ret = trainer.predict(model, test_loader, return_predictions=True)
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


if __name__ == '__main__':
    fire.Fire(train)



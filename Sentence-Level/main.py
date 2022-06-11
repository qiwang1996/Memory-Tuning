import fire
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader
from data import *
from models import AutoModelForSequenceClassificationFinetuner
from sklearn.metrics import accuracy_score, f1_score
from sklearn import preprocessing
from transformers.adapters import MAMConfig, PrefixTuningConfig, AdapterConfig
from ffn_trainable_module import add_ffn_trainable_part, FFNTrainableModuleConfig, \
    freeze_all_parameters, unfreeze_trainable_part
from utils import write_result_to_file, clear_dir, creat_file
import os


def train(
    mode,
    task,
    train_file,
    val_file,
    test_file,
    saved_file,
    preseqlen=None,
    ffn_module_size=None,
    memory_num=None,
    max_length: int = 100,
    model_name: str = './lm/bert-large-uncased',
    num_epochs: int = 50,
    n_workers: int = 1,
    gpus: int = 2,
    precision: int = 32,
    patience: int = 10,
    lr: float = 2e-05,
    batch_size: int = 64,
    layer_remain_id: int = 0,
    random_seed: int = 42
):
    creat_file(saved_file)
    print('------------*Start task:{}, mode:{}, memory_num:{}*------------'.format(task, mode, memory_num))
    torch.random.manual_seed(random_seed)
    task_list = ['MRPC', 'QNLI',
                 'RTE', 'SST-2',  'CB']
    solve_list = [solve_mrpc_data, solve_qnli_data, solve_rte_data,
                  solve_sst_data, solve_cb_data]

    metric_list = [f1_score, accuracy_score, accuracy_score,
                   accuracy_score, accuracy_score]

    assert task in task_list

    solve_func = solve_list[task_list.index(task)]
    eval_func = metric_list[task_list.index(task)]

    label_encoder = preprocessing.LabelEncoder()

    train_dataset, val_dataset, test_dataset, label_encoder = solve_func(train_file, val_file, test_file, label_encoder)

    train_loader = DataLoader(train_dataset, num_workers=n_workers,  
                              shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, num_workers=n_workers,
                            batch_size=batch_size)
    test_loader = DataLoader(test_dataset, num_workers=n_workers,
                             batch_size=batch_size)

    # Load pre-trained model (weights)
    model = AutoModelForSequenceClassificationFinetuner(eval_func, mode,
                                                        model_name,
                                                        n_classes=train_dataset.n_classes,
                                                        preseqlen=preseqlen,
                                                        max_length=max_length,
                                                        lr=lr)

    if mode in ['adapter', 'memory', 'memory-ffn']:
        # Add adapters and freeze all layers
        config = FFNTrainableModuleConfig(
            hidden_size=1024, ffn_module_size=ffn_module_size,
            ffn_module_act='relu', ffn_module_initializer_range=1e-2,
            memory_num=memory_num, ffn_module_layer_norm_eps=1e-12,
        )

        model.model.bert = add_ffn_trainable_part(model.model.bert, config, layer_remain_id)
        model.model.bert = freeze_all_parameters(model.model.bert)

        # Unfreeze adapters and the classifier head
        model.model.bert = unfreeze_trainable_part(model.model.bert)
        model.model.classifier.requires_grad = True
    elif mode == 'prefix':
        model.model.bert = freeze_all_parameters(model.model.bert)
    elif mode == 'mam_adapter':
        prefix_cfg = PrefixTuningConfig(flat=False, prefix_length=preseqlen,
                                        non_linearity='relu', dropout=0.1)
        adapter_cfg = AdapterConfig(
                                    reduction_factor=1024 // ffn_module_size,
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

    dir = os.path.join(os.path.abspath(os.getcwd()), 'saved_model')
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=patience)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath=dir,
                                          filename='model-best-{epoch:02d}-{val_loss:.2f}')

    trainer = pl.Trainer(max_epochs=num_epochs,
                         gpus=gpus,
                         num_processes=4,
                         accelerator='gpu',
                         auto_select_gpus=gpus > 0,
                         auto_scale_batch_size=False,
                         precision=precision,
                         callbacks=[early_stop_callback, checkpoint_callback],
                         )

    trainer.fit(model, train_loader, val_loader)
    
    best_para = torch.load(checkpoint_callback.best_model_path)['state_dict']
    model.load_state_dict(best_para)

    trainer = pl.Trainer(max_epochs=0,
                         gpus=1,
                         auto_scale_batch_size=True,
                         precision=precision)
    ret = trainer.predict(model, test_loader, return_predictions=True)
    write_result_to_file(ret, label_encoder, saved_file)
    clear_dir('./saved_model')
    clear_dir('./lightning_logs')


if __name__ == '__main__':
    fire.Fire(train)



import pandas as pd
from utils import convert_json_file_to_df_data, DataFrameTextClassificationDataset


def solve_sst_data(train_file, val_file, test_file, label_encoder):
    train_df = pd.read_table(train_file, quoting=3)
    val_df = pd.read_table(val_file, quoting=3)
    test_df = pd.read_table(test_file, quoting=3)

    label_encoder.fit(pd.concat([train_df['label'], val_df['label']]))

    train_dataset = DataFrameTextClassificationDataset(train_df, label_encoder,
                                                       x_label='sentence',
                                                       y_label='label',
                                                       )
    val_dataset = DataFrameTextClassificationDataset(val_df, label_encoder,
                                                     x_label='sentence',
                                                     y_label='label')
    test_dataset = DataFrameTextClassificationDataset(test_df, label_encoder,
                                                      x_label='sentence',
                                                      y_label=None)
    return train_dataset, val_dataset, test_dataset, label_encoder


def solve_mrpc_data(train_file, val_file, test_file, label_encoder):
    train_df = pd.read_table(train_file, quoting=3)[['Quality', '#1 String', '#2 String']]
    val_df = pd.read_table(val_file, quoting=3)[['Quality', '#1 String', '#2 String']]
    test_df = pd.read_table(test_file, quoting=3)[['#1 String', '#2 String']]

    label_encoder.fit(pd.concat([train_df['Quality'], val_df['Quality']]))

    for df in [train_df, val_df, test_df]:
        df['sentence'] = '[CLS]' + df['#1 String'] + '[SEP]' + df['#2 String']
        df.drop('#1 String', axis=1, inplace=True)
        df.drop('#2 String', axis=1, inplace=True)

    train_dataset = DataFrameTextClassificationDataset(train_df, label_encoder,
                                                       x_label='sentence',
                                                       y_label='Quality',
                                                       )
    val_dataset = DataFrameTextClassificationDataset(val_df,  label_encoder,
                                                     x_label='sentence',
                                                     y_label='Quality')
    test_dataset = DataFrameTextClassificationDataset(test_df, label_encoder,
                                                      x_label='sentence',
                                                      y_label=None)
    return train_dataset, val_dataset, test_dataset, label_encoder


def solve_qnli_data(train_file, val_file, test_file, label_encoder):
    train_df = pd.read_table(train_file, quoting=3)
    val_df = pd.read_table(val_file, quoting=3)
    test_df = pd.read_table(test_file, quoting=3)

    label_encoder.fit(pd.concat([train_df['label'], val_df['label']]))

    for df in [train_df, val_df, test_df]:
        df['qa'] = '[CLS]' + df['question'] + '[SEP]' + df['sentence']
        df.drop('question', axis=1, inplace=True)
        df.drop('sentence', axis=1, inplace=True)
        df.drop('index', axis=1, inplace=True)

    train_dataset = DataFrameTextClassificationDataset(train_df, label_encoder,
                                                       x_label='qa',
                                                       y_label='label',
                                                       )

    val_dataset = DataFrameTextClassificationDataset(val_df, label_encoder,
                                                     x_label='qa',
                                                     y_label='label')

    test_dataset = DataFrameTextClassificationDataset(test_df, label_encoder,
                                                      x_label='qa',
                                                      y_label=None)
    return train_dataset, val_dataset, test_dataset, label_encoder


def solve_rte_data(train_file, val_file, test_file, label_encoder):
    train_df = pd.read_table(train_file, quoting=3)[['sentence1', 'sentence2', 'label']]
    val_df = pd.read_table(val_file, quoting=3)[['sentence1', 'sentence2', 'label']]
    test_df = pd.read_table(test_file, quoting=3)[['sentence1', 'sentence2']]

    label_encoder.fit(pd.concat([train_df['label'], val_df['label']]))

    for df in [train_df, val_df, test_df]:
        df['sentence'] = '[CLS]' + df['sentence1'] + '[SEP]' + df['sentence2']
        df.drop('sentence1', axis=1, inplace=True)
        df.drop('sentence2', axis=1, inplace=True)

    train_dataset = DataFrameTextClassificationDataset(train_df, label_encoder,
                                                       x_label='sentence',
                                                       y_label='label',
                                                       )

    val_dataset = DataFrameTextClassificationDataset(val_df, label_encoder,
                                                     x_label='sentence',
                                                     y_label='label')

    test_dataset = DataFrameTextClassificationDataset(test_df, label_encoder,
                                                      x_label='sentence',
                                                      y_label=None)
    return train_dataset, val_dataset, test_dataset, label_encoder


def solve_cb_data(train_file, val_file, test_file, label_encoder):
    train_df = convert_json_file_to_df_data(train_file)
    val_df = convert_json_file_to_df_data(val_file)
    test_df = convert_json_file_to_df_data(test_file)

    label_encoder.fit(pd.concat([train_df['label'], val_df['label']]))

    for df in [train_df, val_df, test_df]:
        df['sentence'] = '[CLS]' + df['premise'] + '[SEP]' + df['hypothesis']
        df.drop('premise', axis=1, inplace=True)
        df.drop('hypothesis', axis=1, inplace=True)

    train_dataset = DataFrameTextClassificationDataset(train_df, label_encoder,
                                                       x_label='sentence',
                                                       y_label='label',
                                                       )

    val_dataset = DataFrameTextClassificationDataset(val_df, label_encoder,
                                                     x_label='sentence',
                                                     y_label='label', )

    test_dataset = DataFrameTextClassificationDataset(test_df, label_encoder,
                                                      x_label='sentence',
                                                      y_label=None)

    return train_dataset, val_dataset, test_dataset, label_encoder



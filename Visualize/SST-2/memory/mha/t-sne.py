from sklearn import manifold, datasets
from sklearn.decomposition import PCA
import numpy as np
import json

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def visualized(tensor_dic):
    tensor_li = [(label, tensor) for label, tensor in tensor_dic.items()]
    len_li = [len(tensor) for (label, tensor) in tensor_li]
    labels = [label for (label, tensor) in tensor_li]

    color = ['red', 'blue']

    x = []

    for i, (label, tensor) in enumerate(tensor_li):
        x += tensor

    ts = manifold.TSNE()
    pca = PCA(n_components=10)
    x = pca.fit_transform(x)

    y = ts.fit_transform(x)
    fig = plt.figure(figsize=(15, 7.5), dpi=80)

    s, t = 0, 0
    for i, c in enumerate(color):
        t = s + len_li[i]
        plt.scatter(y[s:t, 0], y[s:t, 1], c=c)
        s = t

    plt.legend(labels=labels, loc="upper right", fontsize=20)
    # plt.title('Dimension reduction clustering analysis results on attention vector', fontsize=20)
    plt.show()


def parse_data(data_file):

    index_dic = json.load(open('index.json', 'r', encoding='utf-8'))
    tensor_dic = {label: [] for label in index_dic.keys()}

    # layer, bs, seq_len, n_memory
    line = [_ for _ in open(data_file, 'r').readlines() if _ != '\n'][0]
    line = line.strip()
    line = line.replace('[', ' ').replace(']', ' ').replace(',', ' ')
    data = np.array(line.split()).astype(float)

    data = np.reshape(data, [872, 12 * 16])
    data = data.tolist()

    for label, index in index_dic.items():
        for ind in index:
            tensor_dic[label].append(data[ind])

    return tensor_dic


attns = parse_data('prefix_attn.txt')
visualized(attns)

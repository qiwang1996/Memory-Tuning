import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn import manifold, datasets
from sklearn.decomposition import PCA
import numpy as np
import json


def visualized(tensor_dic):
    tensor_li = [(label, tensor) for label, tensor in tensor_dic.items()]
    len_li = [len(tensor) for (label, tensor) in tensor_li]
    labels = [label for (label, tensor) in tensor_li]

    color = ['purple', 'blue', 'orange', 'black', 'green', 'red', 'grey']

    x = []
    x_color = []

    for i, (label, tensor) in enumerate(tensor_li):
        if label == 'O':
            x += tensor[:1500]
            x_color += [color[i]] * 1500
            len_li[i] = 1500
        else:
            x += tensor
            x_color += [color[i]] * len(tensor)

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
    # plt.title('Attention Vector Cluster after Dimension Reduction', fontsize=20)
    plt.show()


def parse_data(data_file):
    index_dic = json.load(open('index.json', 'r', encoding='utf-8'))
    tensor_dic = {label: [] for label, _ in index_dic.items()}

    # layer, bs, seq_len, n_memory
    line = [_ for _ in open(data_file, 'r').readlines() if _ != '\n'][0]
    line = line.strip()
    line = line.replace('[', ' ').replace(']', ' ').replace(',', ' ')
    data = np.array(line.split()).astype(float)

    data = np.reshape(data, [2318, 100, 50])
    data = data.tolist()

    for label, index in index_dic.items():
        print(label, ':', len(index))
        limit_len = len(data[0])
        for i, j in index:
            if j + 1 < limit_len:
                tensor_dic[label].append(data[i][j+1])

    return tensor_dic


attns = parse_data('memory_attn.txt')

visualized(attns)

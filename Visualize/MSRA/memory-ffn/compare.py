import json

label = open('label.txt', 'r', encoding='utf-8').readlines()
predict = open('pred.txt', 'r', encoding='utf-8').readlines()

dic = {}
find_labels = ['I-PER', 'B-PER', 'I-ORG', 'B-ORG', 'I-LOC', 'B-LOC', 'O']


for i, (l_line, p_line) in enumerate(zip(label, predict)):
    l = l_line.strip().split()
    p = p_line.strip().split()

    for j, (l_item, p_item) in enumerate(zip(l, p)):
        if l_item in find_labels and l_item == p_item:
            if dic.get(l_item, None) is None:
                dic[l_item] = []
            dic[l_item].append((i, j))

cnt = 0
for k in dic.keys():
    cnt += len(dic[k])
    print(len(dic[k]))


print(cnt)

json.dump(dic, open('index.json', 'w', encoding='utf-8'))




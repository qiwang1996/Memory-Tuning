import json

label = open('dev.tsv', 'r', encoding='utf-8').readlines()
predict = open('SST-2.tsv', 'r', encoding='utf-8').readlines()

print(len(label))

dic = {'Positive': [], 'Negative': []}
find_labels = ['0', '1']


for i, (l_line, p_line) in enumerate(zip(label, predict)):
    if i == 0:
        continue

    l = l_line.strip().split()[-1]
    p = p_line.strip().split()[-1]

    if 1 or l == p:
        if l == '1':
            dic['Positive'].append(i-1)
        else:
            dic['Negative'].append(i-1)

cnt = 0
for k in dic.keys():
    cnt += len(dic[k])
print(cnt)

json.dump(dic, open('index.json', 'w', encoding='utf-8'))




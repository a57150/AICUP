import glob
from collections import defaultdict
import pandas as pd

#====================================================================================================
#抓取第一階段資料集

sentence_map = {}   # {file1: [sentense1, sentense2, ... ], file2: [sentense1, sentense2, ... ], ... }
position_map = {}   # {file1: [(start1, end1), (start2, end2), ... ], file2: [(start1, end1), (start2, end2), ... ], ... }

for fname in glob.glob("/Data/First_Phase_Release(Correction)/First_Phase_Text_Dataset/*.txt"):
    with open(fname) as f:
        name = fname.replace("/Data/First_Phase_Release(Correction)/First_Phase_Text_Dataset/", '').replace('.txt', '')
        sentences = f.readlines()
        sentence_map[name] = sentences

        positions = [(0, 0)] 
        for sents in sentences:
            positions.append((positions[-1][1], positions[-1][1] + len(sents)))
        
        position_map[name] = positions[1:] 


answer_map = defaultdict(list)   # {file1: [label1, label2, ... ], file2: [label1, label2, ... ], ... }

for fname in glob.glob("/Data/First_Phase_Release(Correction)/*.txt"):
    with open(fname, encoding='utf-8-sig') as f:
        answers = f.readlines()
        for ans in answers:
            name = ans.split('\t')[0]
            if ans.split('\t')[1] == 'COUNTRY' or ans.split('\t')[1] == 'LOCATION-OTHER': continue
            answer_map[name].append(ans.split('\t')[1:])

#----------------------------------------------------------------------------------------------------

data = []

for fname in glob.glob("/Data/First_Phase_Release(Correction)/First_Phase_Text_Dataset/*.txt"):
    name = fname.replace("/Data/First_Phase_Release(Correction)/First_Phase_Text_Dataset/", '').replace('.txt', '')
    i = j = 0
    n = len(answer_map[name])

    while j < len(sentence_map[name]):

        contents_start, contents_end = position_map[name][j][0], position_map[name][j][1]  # 紀錄 sentense 的起始位置 及 結束位置

        
        if i < n and int(answer_map[name][i][2]) < contents_start:   # 此 sentense 有超過一個 label
            data[-1][4] = data[-1][4] + '\t'.join(str(item) for item in answer_map[name][i])
            i += 1
        elif i < n and int(answer_map[name][i][1]) >= contents_start and int(answer_map[name][i][2]) < contents_end:   # label 的位置  在 sentense 裡
            data.append([
                name,
                position_map[name][j][0], 
                position_map[name][j][1], 
                sentence_map[name][j].replace("\n", '\t'), 
                '\t'.join(str(item) for item in answer_map[name][i])
            ])
            i += 1
            j += 1
        else:    # 此 sentense 沒有 label
            data.append([
                name,
                position_map[name][j][0], 
                position_map[name][j][1], 
                sentence_map[name][j].replace("\n", '\t'), 
                'PHI: NULL'
            ])
            j += 1

#====================================================================================================
#抓取第二階段資料集

sentence_map = {}
position_map = {}

for fname in glob.glob("/Data/Second_Phase_Dataset/Second_Phase_Text_Dataset/*.txt"):
    with open(fname) as f:
        name = fname.replace("/Data/Second_Phase_Dataset/Second_Phase_Text_Dataset/", '').replace('.txt', '')
        sentences = f.readlines()
        sentence_map[name] = sentences

        positions = [(0, 0)] 
        for sents in sentences:
            positions.append((positions[-1][1], positions[-1][1] + len(sents)))
        
        position_map[name] = positions[1:] 

answer_map = defaultdict(list)

for fname in glob.glob("/Data/Second_Phase_Dataset/*.txt"):
    with open(fname, encoding='utf-8-sig') as f:
        answers = f.readlines()
        for ans in answers:
            name = ans.split('\t')[0]
            if ans.split('\t')[1] == 'COUNTRY' or ans.split('\t')[1] == 'LOCATION-OTHER': continue
            answer_map[name].append(ans.split('\t')[1:])

for fname in glob.glob("/Data/Second_Phase_Dataset/Second_Phase_Text_Dataset/*.txt"):
    name = fname.replace("/Data/Second_Phase_Dataset/Second_Phase_Text_Dataset/", '').replace('.txt', '')
    i = j = 0
    n = len(answer_map[name])

    while j < len(sentence_map[name]):

        contents_start, contents_end = position_map[name][j][0], position_map[name][j][1]

        
        if i < n and int(answer_map[name][i][2]) < contents_start:
            data[-1][4] = data[-1][4] + '\t'.join(str(item) for item in answer_map[name][i])
            i += 1
        elif i < n and int(answer_map[name][i][1]) >= contents_start and int(answer_map[name][i][2]) < contents_end:
            data.append([
                name,
                position_map[name][j][0], 
                position_map[name][j][1], 
                sentence_map[name][j].replace("\n", '\t'), 
                '\t'.join(str(item) for item in answer_map[name][i])
            ])
            i += 1
            j += 1
        else:
            data.append([
                name,
                position_map[name][j][0], 
                position_map[name][j][1], 
                sentence_map[name][j].replace("\n", '\t'), 
                'PHI: NULL'
            ])
            j += 1

#====================================================================================================
#讀取 ChatGPT 擴增的資料集

with open("/Data/duration.txt") as f:
    cont = f.readlines()
    i = 0
    for c in cont:
        if i == 0:
            content = c
            i += 1
        elif i == 1:
            label = c
            i += 1
        else:
            data.append([
                'name',
                0, 
                0, 
                content.replace("\n", '\t'), 
                label
            ])
            i = 0

with open("/Data/phone.txt") as f:
    cont = f.readlines()
    i = 0
    for c in cont:
        if i == 0:
            content = c
            i += 1
        elif i == 1:
            label = c
            i += 1
        else:
            data.append([
                'name',
                0, 
                0, 
                content.replace("\n", '\t'), 
                label
            ])
            i = 0

#====================================================================================================
#儲存資料集

df = pd.DataFrame(data, columns=['file', 'contents_start', 'contents_end', 'contents', 'labels'])
csv_filename = '/Data/train_datas.csv'
df.to_csv(csv_filename, index=False)

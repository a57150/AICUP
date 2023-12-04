#====================================================================================================

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

plm = "EleutherAI/pythia-160m-deduped"

bos = '<|endoftext|>'
eos = '<|END|>'
pad = '<|pad|>'
sep = '\n\n####\n\n'

special_tokens_dict = {'eos_token': eos, 'bos_token': bos, 'pad_token': pad, 'sep_token': sep}

tokenizer = AutoTokenizer.from_pretrained(plm, revision="step3000")
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)


PAD_IDX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
IGNORED_PAD_IDX = -100

import torch
from torch.optim import AdamW
model = AutoModelForCausalLM.from_pretrained(plm, revision='step3000')
EPOCHS = 5
optimizer = AdamW(model.parameters(), lr=5e-5)
model.resize_token_embeddings(len(tokenizer))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

from peft import LoraConfig, TaskType
from peft import get_peft_model

peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)

model = get_peft_model(model, peft_config)

optimizer = AdamW(model.parameters(), lr=5e-5)
model.resize_token_embeddings(len(tokenizer))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.print_trainable_parameters()
model.to(device)

#====================================================================================================

model.load_state_dict(torch.load('models\\peft5_epo6_pythia_160md.pt'))
model = model.merge_and_unload()

#====================================================================================================

from datasets import load_dataset, Features, Value
valid_data = load_dataset("csv", data_files="AICUP\\opendid_test.tsv", delimiter='\t',
                          features = Features({
                              'fid': Value('string'),
                              'idx': Value('int64'),
                              'content': Value('string')}),
                              column_names=['fid', 'idx', 'content'])
valid_list = list(valid_data['train'])

#====================================================================================================

from torch.nn import functional as F
import re
import torch

dic = [ 'PATIENT', 'DOCTOR', 'USERNAME', 'PROFESSION', 'ROOM', 'DEPARTMENT', 'HOSPITAL', 'ORGANIZATION', 'STREET', 'CITY', 'STATE', 'COUNTRY', 'ZIP', 'LOCATION-OTHER', 'AGE', 'DATE',
    'TIME', 'DURATION', 'SET', 'PHONE', 'FAX', 'EMAIL', 'URL', 'IPADDR', 'SSN', 'MEDICALRECORD', 'HEALTHPLAN', 'ACCOUNT', 'LICENSE', 'VECHICLE', 'DEVICE', 'BIOID', 'IDNUM', 'OTHER']

du = r'P\d\d?(\.\d\d?)?W|P\d\d?(\.\d\d?)?K|P\d\d?(\.\d\d?)?Y|P\d\d?(\.\d\d?)?D|P\d\d?(\.\d\d?)?M'

def predict(model, tokenizer, input, max_new_tokens=400):

    #-------------------------------------------------------------------
    def modify(string, file, start, input):  
        nonlocal end

        #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        def found_position(string, found, start):   #找出擷取的資料在哪個位置
            nonlocal end
            position = string.find(found, end)

            if position != -1:
                end = position + len(found)
                res = f'{start+position}\t{start+position+len(found)}'
                return res
        #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

        pattern = re.compile(r'(\w+)\t(.+)')
        content = re.sub(pattern, r'\2', string)
        PHI = re.sub(pattern, r'\1', string)

        if PHI not in dic:
            return ""
        
        #對有正規化的資料額外做處理
        if PHI == 'DATE':
            cont = content.split('\t')
            start_end = found_position(input, cont[0], int(start))
            if len(cont) >= 2:
                if len(cont[1]) > 10:
                    cont[1] = cont[1][:10]
                content = cont[0] + '\t' + cont[1]
            else:
              return ""
    
        elif PHI == 'TIME':
            cont = content.split('\t')
            start_end = found_position(input, cont[0], int(start))
            if len(cont) >= 2:
                if len(cont[1]) > 16:
                    cont[1] = cont[1][:16]
                content = cont[0] + '\t' + cont[1]
            else:
              return ""
        elif PHI == 'DURATION':
            cont = content.split('\t')
            start_end = found_position(input, cont[0], int(start))
            if len(cont) >= 2:
                if len(cont[1]) > 4:
                    cont[1] = cont[1][:4]
                if not re.match(du, cont[1]):
                  return ""
                content = cont[0] + '\t' + re.match(du, cont[1]).group()
            else:
              return ""
        else:
            start_end = found_position(input, content, int(start))

        if start_end is None:
            return ""

        modified = re.sub(pattern, r'{}\t\1\t{}\t\2'.format(file, start_end), PHI + '\t' + content)
        return modified
    #-------------------------------------------------------------------

    model.eval()
    input = input.split('<-#-#-#->')
    prompt = bos + input[2] + sep   #input[2] 為文章內容
    end = 0   #紀錄 label(擷取到的答案資訊) 的最後位置

    tks_info = tokenizer(prompt)
    text = tks_info['input_ids']

    inputs, past_key_values = torch.tensor([text]), None
    outputs = ''

    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(inputs.to(device), past_key_values=past_key_values)
            logits = out.logits
            past_key_values = out.past_key_values
            log_probs = F.softmax(logits[:, -1], dim=-1)
            inputs = torch.argmax(log_probs, 1).unsqueeze(0)

            if tokenizer.decode(inputs.item()) == eos:
                break
            text.append(inputs.item())

        pred = tokenizer.decode(text)
        pred = pred[pred.find(sep) + len(sep):].replace(pad, "").replace(eos, "").strip()

        if pred.split('\t')[-1] == 'PHI: NULL':   #如果沒擷取到資訊
            return ""

        labels = pred.split('\n')   #擷取到的資訊可能不只有一個

        for i in range(len(labels)):
            if i < len(labels) - 1:
                if len(labels[i].split("\t")) < 2:
                    continue
                o = modify(labels[i], input[0], input[1], input[2])
                if o == "":
                    continue
                outputs += o
                if end > len(input[2]):
                    break
                outputs += '\n'
            else:
                if len(labels[i].split("\t")) < 2:
                    continue
                o = modify(labels[i], input[0], input[1], input[2])
                if o == "":
                    continue
                outputs += o

        outputs = outputs.rstrip('\n')

    return outputs

#====================================================================================================

answer = []

lo = r'P\.?\ ?O\.?\ ?BOX \d+'
co = ['South Africa', 'Vietnam', 'Australia']

for sent in valid_list:

    input_ = sent['content']
    if input_ is None: continue
    ans = predict(model, tokenizer, sent['fid']+'<-#-#-#->'+str(sent['idx'])+'<-#-#-#->'+input_)

    for c in co:  #抓出COUNTRY
        if ans.find('ZIP') > -1: break
        pos1 = input_.find(c)
        if pos1 > -1:
            pos1 += sent['idx']
            if ans is None or ans == "":
                ans = sent['fid']+'\t'+'COUNTRY'+'\t'+str(pos1)+'\t'+str(pos1+len(c))+'\t'+c
                break
            a1 = sent['idx']
            a2 = ans.split('\n')
            for i in range(len(a2)):
                if pos1+len(c) <= int(a2[i].split('\t')[2]) and pos1 >= a1:
                    a2.insert(i, sent['fid']+'\t'+'COUNTRY'+'\t'+str(pos1)+'\t'+str(pos1+len(c))+'\t'+c)
                    ans = '\n'.join(a2)
                    break
                a1 = int(a2[i].split('\t')[3])
            a1 = int(a2[-1].split('\t')[3])
            if pos1 >= a1:
                print(a2)
                a2.append(sent['fid']+'\t'+'COUNTRY'+'\t'+str(pos1)+'\t'+str(pos1+len(c))+'\t'+c)
                ans = '\n'.join(a2)
                print(a2)
                break

            for j in range(len(a2)):
                if c == a2[j].split('\t')[-1]:
                    a2[j] = sent['fid']+'\t'+'COUNTRY'+'\t'+str(pos1)+'\t'+str(pos1+len(c))+'\t'+c
                    ans = '\n'.join(a2)
                    break


        match = re.search(lo, input_)
        if match:   #抓出LOCATION-OTHER
            loc = match.group()
            pos = input_.find(loc)+sent['idx']
            if ans is None or ans == "":
                ans = sent['fid']+'\t'+'LOCATION-OTHER'+'\t'+str(pos)+'\t'+str(pos+len(loc))+'\t'+loc
                print(ans)
                answer.append(str(ans)+'\n')
                continue
        a1 = sent['idx']
        a2 = ans.split('\n')
        for i in range(len(a2)):
            if pos+len(loc) <= int(a2[i].split('\t')[2]) and pos >= a1:
                a2.insert(i, sent['fid']+'\t'+'LOCATION-OTHER'+'\t'+str(pos)+'\t'+str(pos+len(loc))+'\t'+loc)
                ans = '\n'.join(a2)
                break
            a1 = int(a2[i].split('\t')[3])

    if ans is None or ans == "": continue
    print(ans)
    answer.append(str(ans)+'\n')

with open('/content/drive/MyDrive/AICUP/answer.txt', 'w') as file:
    file.writelines(answer)

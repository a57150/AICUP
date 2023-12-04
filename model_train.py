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

#====================================================================================================

PAD_IDX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
IGNORED_PAD_IDX = -100
print(PAD_IDX)

#====================================================================================================

import pandas as pd

files="train_datas.csv"
datas = pd.read_csv(files)

#====================================================================================================

from torch.utils.data import DataLoader
import torch
import re

def modify_string(original_string):
    pattern = re.compile(r'(\w+)\t(\d+)\t(\d+)\t(.+)')
    modified_string = re.sub(pattern, r'\1\t\4', original_string)
    return modified_string

train_data = [bos + data['contents'] + sep + modify_string(data['labels']) + eos for _, data in datas.iterrows()]
print(train_data[-1])
print("============================================")
print(len(train_data))

def collate_batch(batch):

    texts = batch

    encoded_seq = tokenizer(texts, padding=True)

    indexed_tks = torch.tensor(encoded_seq['input_ids'])
    attention_mask = torch.tensor(encoded_seq['attention_mask'])
    encoded_label = torch.tensor(encoded_seq['input_ids'])

    encoded_label[encoded_label == tokenizer.pad_token_id] = IGNORED_PAD_IDX

    return indexed_tks, encoded_label, attention_mask

train_dataloader = DataLoader(train_data, batch_size=2, shuffle=False, collate_fn=collate_batch)
titer = iter(train_dataloader)
tks, labels, mask = next(titer)
print(tks.shape)
print(next(iter(titer)))

#====================================================================================================

import random
BATCH_SIZE = 8

class BatchSampler():
    def __init__(self, data, batch_size):
        self.pooled_indice = []
        self.data = data
        self.batch_size = batch_size
        self.len = len(list(data))
    def __iter__(self):
        self.pooled_indices = []
        indices = [(index, len(data)) for index, data in enumerate(self.data)]
        random.shuffle(indices)
        for i in range(0, len(indices), BATCH_SIZE * 100):
            self.pooled_indices.extend(sorted(indices[i:i + BATCH_SIZE * 100], key=lambda x: x[1], reverse=True))
        self.pooled_indices = [x[0] for x in self.pooled_indices]

        for i in range(0, len(self.pooled_indices), BATCH_SIZE):
            yield self.pooled_indices[i:i + BATCH_SIZE]
    def __len__(self):
        return (self.len + self.batch_size - 1) // self.batch_size

bucket_train_dataloader = DataLoader(train_data, batch_sampler=BatchSampler(train_data, BATCH_SIZE),
                                     collate_fn=collate_batch, pin_memory=True)

#====================================================================================================

import torch
from torch.optim import AdamW
model = AutoModelForCausalLM.from_pretrained(plm, revision='step3000')
EPOCHS = 5
optimizer = AdamW(model.parameters(), lr=5e-5)
model.resize_token_embeddings(len(tokenizer))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#====================================================================================================

from tqdm import tqdm, trange

model.train()
for _ in trange(EPOCHS, desc="Epoch"):
    model.train()
    total_loss = 0

    predictions, true_labels = [], []

    for step, (seqs, labels, masks) in enumerate(bucket_train_dataloader):
        seqs = seqs.to(device)
        labels = labels.to(device)
        masks = masks.to(device)
        model.zero_grad()
        outputs = model(seqs, labels=labels)
        logits = outputs.logits
        loss = outputs.loss
        loss = loss.mean()

        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    avg_train_loss = total_loss / len(bucket_train_dataloader)
    print("Average train loss: {}".format(avg_train_loss))

#====================================================================================================

from tqdm import tqdm, trange

model.train()
EPOCHS = 1
for _ in trange(EPOCHS, desc="Epoch"):
    model.train()
    total_loss = 0

    predictions, true_labels = [], []

    for step, (seqs, labels, masks) in enumerate(bucket_train_dataloader):
        seqs = seqs.to(device)
        labels = labels.to(device)
        masks = masks.to(device)
        model.zero_grad()
        outputs = model(seqs, labels=labels)
        logits = outputs.logits
        loss = outputs.loss
        loss = loss.mean()

        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    avg_train_loss = total_loss / len(bucket_train_dataloader)
    print("Average train loss: {}".format(avg_train_loss))

#====================================================================================================

from torch.optim import AdamW
import torch
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

from tqdm import tqdm, trange

EPOCHS = 5
model.train()
for _ in trange(EPOCHS, desc="Epoch"):
    model.train()
    total_loss = 0

    predictions, true_labels = [], []

    for step, (seqs, labels, masks) in enumerate(bucket_train_dataloader):
        seqs = seqs.to(device)
        labels = labels.to(device)
        masks = masks.to(device)
        model.zero_grad()
        outputs = model(seqs, attention_mask=masks, labels=labels)
        logits = outputs.logits
        loss = outputs.loss
        loss = loss.mean()

        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    avg_train_loss = total_loss / len(bucket_train_dataloader)
    print("Average train loss: {}".format(avg_train_loss))

#====================================================================================================

torch.save(model.state_dict(), 'models\\peft5_epo6_pythia_160md.pt')

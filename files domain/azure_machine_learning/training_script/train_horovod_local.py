import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os, argparse, time, random

###############
#remote
###############

'''
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import DistilBertTokenizer
from transformers import DistilBertForSequenceClassification, AdamW, DistilBertConfig
from transformers import get_linear_schedule_with_warmup
import horovod.torch as hvd

from azureml.core import Workspace, Run, Dataset




parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, dest='dataset_name', default='')
parser.add_argument('--batch_size', type=int, dest='batch_size', default=32)
parser.add_argument('--learning_rate', type=float, dest='learning_rate', default=1e-5)
parser.add_argument('--adam_epsilon', type=float, dest='adam_epsilon', default=1e-8)
parser.add_argument('--num_epochs', type=int, dest='num_epochs', default=5)



args = parser.parse_args()

dataset_name = args.dataset_name
batch_size = args.batch_size
learning_rate = args.learning_rate
adam_epsilon = args.adam_epsilon
num_epochs = args.num_epochs


run = Run.get_context()
workspace = run.experiment.workspace

dataset = Dataset.get_by_name(workspace, name=dataset_name)

file_name = dataset.download()[0]

# for original data: CSV
#df = pd.read_csv(file_name)
# for files doamin data : tsv
df = pd.read_csv(file_name, sep='\t', encoding="utf-8")
'''

###############
#local
###############

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import DistilBertTokenizer
from transformers import DistilBertForSequenceClassification, AdamW, DistilBertConfig
from transformers import get_linear_schedule_with_warmup
#import horovod.torch as hvd


# ouput only three column
#df = pd.read_csv('E:/azure_ml_notebook/azureml_data/complaints_after.tsv', sep='\t', encoding="utf-8")
#df = pd.read_csv('E:/azure_ml_notebook/azureml_data/files_domain_training_contexual_answer_small.tsv', sep='\t', encoding="utf-8")
df = pd.read_csv('E:/azure_ml_notebook/azureml_data/files_domain_training_contexual_answer.tsv', sep='\t', encoding="utf-8")
#df = pd.read_csv('E:/azure_ml_notebook/azureml_data/complaints_sampled_after.csv', encoding="utf-8")


# for debug
print('top head data {}'.format(df.head()))


# old data format
'''
label_counts = pd.DataFrame(df['Product'].value_counts())
label_values = list(label_counts.index)
order = list(pd.DataFrame(df['Product_Label'].value_counts()).index)
label_values = [l for _,l in sorted(zip(order, label_values))]

texts = df['Complaint'].values
labels = df['Product_Label'].values
'''

# new format
# label_counts  / label values are useless unless treating them as features
#label_counts = pd.DataFrame(df['Product'].value_counts())
#label_values = list(label_counts.index)
#order = list(pd.DataFrame(df['Product_Label'].value_counts()).index)
#label_values = [l for _,l in sorted(zip(order, label_values))]



###############
#local above
###############





texts = df['query'].values
labels = df['domain'].values

# for debug
# label_counts  / label values are useless unless treating them as features
#print('label_counts {}'.format(label_counts))
#print('label_values after sorted {}'.format(label_values))
print('labels {}'.format(labels))



#also huggingface
#https://huggingface.co/transformers/model_doc/distilbert.html
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)

# for debug
print('Original Text: {}'.format(texts[0]))
print('Tokenized Text: {}'.format(tokenizer.tokenize(texts[0])))
print('Token IDs: {}'.format(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(texts[0]))))

#https://github.com/huggingface/transformers/issues/5397
# doing like this
#encode it to the corresponding numeric values for each token.
#truncate it to the maximum sequence length of 300.
#pad the tokens positions greater than 300.
#include the special token IDs to mark the beginning and end of each sequence.
#text_ids = [tokenizer.encode(text, max_length=300, pad_to_max_length=True) for text in texts]
# remove warning message
#text_ids = [tokenizer.encode(text, max_length=300, pad_to_max_length=True, truncation=True) for text in texts]
# replace with suggested parpamter
text_ids = [tokenizer.encode(text, max_length=300, padding='max_length', truncation=True) for text in texts]


#To fine-tune our model, we need two inputs: one array of token IDs (created above) 
#and one array of a corresponding binary mask, called attention mask in the BERT model specification. 
# Each attention mask has the same length of the corresponding input sequence and has a 0 if 
# the corresponding token is a pad token, or a 1 otherwise.
# length is the same text_ids length (300 setup here)
att_masks = []
for ids in text_ids:
    # if id > 0 , then element will 1
    # otherwise, element will 0
    masks = [int(id > 0) for id in ids]
    att_masks.append(masks)

#sklearn split data
#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
#? same random state but in different ways, how to make sure each query is aligned
#  https://www.cnblogs.com/Yanjy-OnlyOne/p/11288098.html
# it seems same random_state will generate the same result
train_x, test_val_x, train_y, test_val_y = train_test_split(text_ids, labels, random_state=111, test_size=0.2)
train_m, test_val_m = train_test_split(att_masks, random_state=111, test_size=0.2)
test_x, val_x, test_y, val_y = train_test_split(test_val_x, test_val_y, random_state=111, test_size=0.5)
test_m, val_m = train_test_split(test_val_m, random_state=111, test_size=0.5)

#https://pytorch.org/docs/stable/tensors.html
# can be multiple dimentionas
train_x = torch.tensor(train_x)
test_x = torch.tensor(test_x)
val_x = torch.tensor(val_x)
train_y = torch.tensor(train_y)
test_y = torch.tensor(test_y)
val_y = torch.tensor(val_y)
train_m = torch.tensor(train_m)
test_m = torch.tensor(test_m)
val_m = torch.tensor(val_m)

# for debug
# small traninging , 585 query one head
# 585 = 468 + 58 + 59
# 300 is the hard requrement for token length
#train_x dimen torch.Size([468, 300])
#test_x dimen torch.Size([58, 300])
#train_x dimen torch.Size([468, 300])
#val_x dimen torch.Size([59, 300])
#train_y dimen torch.Size([468])
#test_y dimen torch.Size([58])
#train_y dimen torch.Size([468])
#val_y dimen torch.Size([59])
#train_m dimen torch.Size([468, 300])
#test_m dimen torch.Size([58, 300])
#train_m dimen torch.Size([468, 300])
#val_m dimen torch.Size([59, 300])
print('train_x dimen {}'.format(train_x.shape))
print('test_x dimen {}'.format(test_x.shape))
print('train_x dimen {}'.format(train_x.shape))
print('val_x dimen {}'.format(val_x.shape))

print('train_y dimen {}'.format(train_y.shape))
print('test_y dimen {}'.format(test_y.shape))
print('train_y dimen {}'.format(train_y.shape))
print('val_y dimen {}'.format(val_y.shape))

print('train_m dimen {}'.format(train_m.shape))
print('test_m dimen {}'.format(test_m.shape))
print('train_m dimen {}'.format(train_m.shape))
print('val_m dimen {}'.format(val_m.shape))


###############
#training data setup in learning 
###############

'''
# kwargs = {'num_workers': 1, 'pin_memory': True} if gpu_available else {}

#https://zhuanlan.zhihu.com/p/76638962
#hvd distributed learning
hvd.init()

train_data = TensorDataset(train_x, train_m, train_y)
train_sampler = DistributedSampler(train_data, num_replicas=hvd.size(), rank=hvd.rank())
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

val_data = TensorDataset(val_x, val_m, val_y)
val_sampler = DistributedSampler(val_data, num_replicas=hvd.size(), rank=hvd.rank())
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

gpu_available = torch.cuda.is_available()

if gpu_available:
    print("gpu_availabe: {}".format(gpu_available))
    torch.cuda.set_device(hvd.local_rank())

num_labels = len(set(labels))

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels,
                                                            output_attentions=False, output_hidden_states=False)

lr_scaler = hvd.size()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.2},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
compression = hvd.Compression.fp16
optimizer = hvd.DistributedOptimizer(optimizer,
                                     named_parameters=model.named_parameters(),
                                     compression=compression)
# optimizer = hvd.DistributedOptimizer(optimizer,
#                                      named_parameters=model.named_parameters())

total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()

seed_val = 111
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

train_losses = []
val_losses = []
num_mb_train = len(train_dataloader)
num_mb_val = len(val_dataloader)

if num_mb_val == 0:
    num_mb_val = 1

for n in range(num_epochs):
    train_loss = 0
    val_loss = 0
    start_time = time.time()
    
    for k, (mb_x, mb_m, mb_y) in enumerate(train_dataloader):
        optimizer.zero_grad()
        model.train()
        
        mb_x = mb_x.to(device)
        mb_m = mb_m.to(device)
        mb_y = mb_y.to(device)
        
        outputs = model(mb_x, attention_mask=mb_m, labels=mb_y)
        
        loss = outputs[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        train_loss += loss.data / num_mb_train
    
    print ("\nTrain loss after itaration %i: %f" % (n+1, train_loss))
    avg_train_loss = metric_average(train_loss, 'avg_train_loss')
    print ("Average train loss after iteration %i: %f" % (n+1, avg_train_loss))
    train_losses.append(avg_train_loss)
    
    with torch.no_grad():
        model.eval()
        
        for k, (mb_x, mb_m, mb_y) in enumerate(val_dataloader):
            mb_x = mb_x.to(device)
            mb_m = mb_m.to(device)
            mb_y = mb_y.to(device)
        
            outputs = model(mb_x, attention_mask=mb_m, labels=mb_y)
            
            loss = outputs[0]
            
            val_loss += loss.data / num_mb_val
            
        print ("Validation loss after itaration %i: %f" % (n+1, val_loss))
        avg_val_loss = metric_average(val_loss, 'avg_val_loss')
        print ("Average validation loss after iteration %i: %f" % (n+1, avg_val_loss))
        val_losses.append(avg_val_loss)
    
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f'Time: {epoch_mins}m {epoch_secs}s')

if hvd.rank() == 0:
    
    out_dir = './outputs'
    
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    with open(out_dir + '/train_losses.pkl', 'wb') as f:
        joblib.dump(train_losses, f)

    with open(out_dir + '/val_losses.pkl', 'wb') as f:
        joblib.dump(val_losses, f)

    run.log('validation loss', avg_val_loss)
'''
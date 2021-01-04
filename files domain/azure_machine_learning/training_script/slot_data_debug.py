# source code from here
#https://github.com/vilcek/fine-tuning-BERT-for-text-classification/blob/master/02-data-classification.ipynb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os, argparse, time, random
import random;
import re;

###############
#remote
###############

'''
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import DistilBertTokenizer,DistilBertTokenizerFast
from transformers import DistilBertForTokenClassification, AdamW, DistilBertConfig
from transformers import get_linear_schedule_with_warmup
from transformers import BatchEncoding
from tokenizers import Encoding
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


# for debug
print('top head data {}'.format(df.head()))
'''

###############
#local below
###############


import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import DistilBertTokenizer,DistilBertTokenizerFast
from transformers import DistilBertForTokenClassification, AdamW, DistilBertConfig
from transformers import get_linear_schedule_with_warmup
from transformers import BatchEncoding
from tokenizers import Encoding

#import horovod.torch as hvd

#from azureml.core import Workspace, Run, Dataset

# ouput only three column
df = pd.read_csv('E:/azure_ml_notebook/azureml_data/files_slot_training_small.tsv', sep='\t', encoding="utf-8")
#df = pd.read_csv('E:/azure_ml_notebook/azureml_data/files_slot_training_single.tsv', sep='\t', encoding="utf-8")
df = pd.read_csv('E:/azure_ml_notebook/azureml_data/files_slot_training.tsv', sep='\t', encoding="utf-8")


# for debug
print('top head data {}'.format(df.head()))



#print('query nan :{}'.format(df[df['query'].isna()]))
#print('QueryXmluery nan :{}'.format(df[df['QueryXml'].isna()]))



# old data format
#label_counts = pd.DataFrame(df['Product'].value_counts())
#label_values = list(label_counts.index)
#order = list(pd.DataFrame(df['Product_Label'].value_counts()).index)
#label_values = [l for _,l in sorted(zip(order, label_values))]

#texts = df['Complaint'].values
#labels = df['Product_Label'].values

# new format
# label_counts  / label values are useless unless treating them as features
#label_counts = pd.DataFrame(df['Product'].value_counts())
#label_values = list(label_counts.index)
#order = list(pd.DataFrame(df['Product_Label'].value_counts()).index)
#label_values = [l for _,l in sorted(zip(order, label_values))]


###############
#local above
###############

#texts = df['query'].values


#labels = df['QueryXml'].values

### for debug
### label_counts  / label values are useless unless treating them as features
###print('label_counts {}'.format(label_counts))
###print('label_values after sorted {}'.format(label_values))
##print('labels {}'.format(labels))


# read label
from typing_extensions import TypedDict
from typing import List,Any
IntList = List[int] # A list of token_ids
IntListList = List[IntList] # A List of List of token_ids, e.g. a Batch


import itertools
class LabelSet:
    def __init__(self, labels: List[str]):
        self.labels_to_id = {}
        self.ids_to_label = {}

        self.labels_to_id["o"] = 0
        self.ids_to_label[0] = "o"
        num = 1
        for label in labels:
            if label == "o":
                print("skip:{}".format(label))
                continue
            self.labels_to_id[label] = num
            self.ids_to_label[num] = label
            num = num +1 


    def get_aligned_label_ids_from_aligned_label(self, aligned_labels):
        return list(map(self.labels_to_id.get, aligned_labels))

    def get_untagged_id(self):
        return self.labels_to_id["o"]

    def get_labels(self):
        return self.labels_to_id

slots = ["O", 
    "file_name", 
    "file_type", 
    "data_source", 
    "contact_name", 
    "to_contact_name",
    "file_keyword",
    "date",
    "time",
    "meeting_starttime",
    "file_action",
    "file_action_context",
    "position_ref",
    "order_ref",
    "file_recency",
    "sharetarget_type",
    "sharetarget_name",
    "file_folder",
    "data_source_name",
    "data_source_type",
    "attachment"]

# map all slots to lower case
slots_label_set = LabelSet(labels=map(str.lower,slots))



#also huggingface
#https://huggingface.co/transformers/model_doc/distilbert.html
#tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
fast_tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased') # Load a pre-trained tokenizer

# for debug
##print('Original Text: {}'.format(texts[0]))
##print('Tokenized Text: {}'.format(tokenizer.tokenize(texts[0])))
##print('Token IDs: {}'.format(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(texts[0]))))

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
#text_ids = [tokenizer.encode(text, max_length=300, padding='max_length', truncation=True) for text in texts]

text_ids = []
att_masks = []
# iterative get labele and also append padding based on text_ids
labels_for_text_ids = []
for i, row in df.iterrows():
    


    text = row['query']


    text_id = fast_tokenizer.encode(text, max_length=300, padding='max_length', truncation=True)
    text_ids.append(text_id)

    slot = row['QueryXml']
	# remove head and end spaces 
    slot = slot.strip()


    # for debug
    #print("text:{}".format(text))
    #print("text id :{}".format(text_id))
    #print("slot: {}".format(slot))



    annotations = []

    # for contact_name to reanme to to_contact_name
    xmlpairs = re.findall("(<.*?>.*?<\/.*?>)", slot)

    textIndex = 0
    for xmlpair in xmlpairs:
        # extra type and value for xml tag
        xmlTypeEndInd = xmlpair.find(">")

        xmlType = xmlpair[1:xmlTypeEndInd]

        xmlValue = xmlpair.replace("<"+xmlType+">", "")
        xmlValue = xmlValue.replace("</"+xmlType+">", "")
        xmlValue = xmlValue.strip()

        start = text.lower()[textIndex:].find(xmlValue.lower())
        if start == -1:
            #print("skipped text: {} and pair: {}".format(row['query'], xmlValue))
            continue
        # update textIndex according to moving order
        textIndex = start

        annotations.append(dict(start=start,end=start+len(xmlValue),text=xmlValue,label=xmlType))

    fast_tokenized_batch : BatchEncoding = fast_tokenizer(text)
    fast_tokenized_text :Encoding  =fast_tokenized_batch[0]

    # fast token will add CLS and SEP 
    # ? not sure in real trainnig , do we need to provide or not
    # in yue case it does not include those two
    #print("fast token ouput: {}".format(fast_tokenized_text.tokens))

    tokens = fast_tokenized_text.tokens
    aligned_labels = ["O"]*len(tokens) # Make a list to store our labels the same length as our tokens
    for anno in (annotations):
        for char_ix in range(anno['start'],anno['end']):
            token_ix = fast_tokenized_text.char_to_token(char_ix)
            if token_ix is not None: # White spaces have no token and will return None
                aligned_labels[token_ix] = anno['label']


    # for debug
    #for token,label in zip(tokens,aligned_labels):
    #    print (token,"-",label) 

    aligned_label_ids = slots_label_set.get_aligned_label_ids_from_aligned_label(
        map(str.lower,aligned_labels)
    )

    # for debug
    #for token, label in zip(tokens, aligned_label_ids):
    #    print(token, "-", label)


    # append mask
    masks = [int(id > 0) for id in text_id]
    att_masks.append(masks)


    # append align label by following text_ids's length (wih padding)
    labels_for_text_id = []
    for i in range(0,len(text_id)):
        if i < len(aligned_label_ids):
            labels_for_text_id.append(aligned_label_ids[i])
        else:
            # padding, add label as zero as default
            labels_for_text_id.append(slots_label_set.get_untagged_id())
    labels_for_text_ids.append(labels_for_text_id)

    # for debug
    try:
        train_x = torch.tensor(text_id)
        train_y = torch.tensor(labels_for_text_id)
    except (RuntimeError):
        print('torch sensor error query: {}'.format(row['query']))
        print('torch sensor error xml: {}'.format(row['QueryXml']))

### for debug

#print('text_ids[0]: {}'.format(text_ids[0]))
#print('labels_for_text_ids[0]: {}'.format(labels_for_text_ids[0]))
#print('att mask[0] : {}'.format(att_masks[0]))

#print('text_ids: {}'.format(len(text_ids)))
#print('text_ids: {}'.format(len(text_ids[0])))
#print('labels_for_text_ids: {}'.format(len(labels_for_text_ids)))
#print('labels_for_text_ids: {}'.format(len(labels_for_text_ids[0])))
#print('att mask : {}'.format(len(att_masks)))
#print('att mask : {}'.format(len(att_masks[0])))



#sklearn split data
#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
#? same random state but in different ways, how to make sure each query is aligned
#  https://www.cnblogs.com/Yanjy-OnlyOne/p/11288098.html
# it seems same random_state will generate the same result
train_x, test_val_x, train_y, test_val_y = train_test_split(text_ids, labels_for_text_ids, random_state=111, test_size=0.2)
train_m, test_val_m = train_test_split(att_masks, random_state=111, test_size=0.2)
test_x, val_x, test_y, val_y = train_test_split(test_val_x, test_val_y, random_state=111, test_size=0.5)
test_m, val_m = train_test_split(test_val_m, random_state=111, test_size=0.5)




# Convert all inputs and labels into torch tensors, the required datatype 
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


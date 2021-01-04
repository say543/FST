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


import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers.modeling_outputs import TokenClassifierOutput
from transformers import DistilBertPreTrainedModel, DistilBertModel
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


###############
#local below
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

#import horovod.torch as hvd

#from azureml.core import Workspace, Run, Dataset

# ouput only three column
df = pd.read_csv('E:/azure_ml_notebook/azureml_data/files_slot_training_small.tsv', sep='\t', encoding="utf-8")
#df = pd.read_csv('E:/azure_ml_notebook/azureml_data/files_slot_training_single.tsv', sep='\t', encoding="utf-8")


# for debug
print('top head data {}'.format(df.head()))


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
'''

###############
#local above
###############

##texts = df['query'].values


##labels = df['domain'].values

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
            print("skipped text: {} and pair: {}".format(row['query'], xmlValue))
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



### for debug
print('text_ids[0]: {}'.format(text_ids[0]))
print('labels_for_text_ids[0]: {}'.format(labels_for_text_ids[0]))
print('att mask[0] : {}'.format(att_masks[0]))


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


###############
#training data setup in learning 
###############


# kwargs = {'num_workers': 1, 'pin_memory': True} if gpu_available else {}

#https://zhuanlan.zhihu.com/p/76638962
#buer distributed learning
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

num_labels = len(set(slots_label_set.get_labels()))


#Here we instantiate our model class. 
#We use a compact version, that is trained through model distillation from a base BERT model and modified to include a classification layer at the output. This compact version has 6 transformer layers instead of 12 as in the original BERT model.
# this class class DistilBertForSequenceClassification(DistilBertPreTrainedModel):
# 
#https://huggingface.co/transformers/v1.2.0/_modules/pytorch_transformers/modeling_distilbert.html
#model = DistilBertForTokenClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels,
#                                                            output_attentions=False, output_hidden_states=False)

class DistilBertForTokenClassificationFilesDomain(DistilBertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertForSequenceClassificationFilesDomain.from_pretrained('distilbert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
    """

    def __init__(self, config, weight=None):
        super(DistilBertForTokenClassificationFilesDomain, self).__init__(config)
        self.num_labels = config.num_labels
        self.weight = weight

        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        self.init_weights()

        '''
        super().__init__(config)
        self.num_labels = config.num_labels

        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()
        '''
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # this line is related to wrong message
        outputs = self.distilbert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # originla value
        #output = (logits,) + outputs[1:]
        #return ((loss,) + output) if loss is not None else output

        #version1
        #slot_label_id = []
        #for i, ele2d in enumerate(logits):
        #    for j, ele1d in enumerate(ele2d):
        #        value, label_id = torch.max(ele1d, dim=0)
        #        slot_label_id.append(label_id)
        #        #print("value: {} anmd index {}".format(value, index))

        #slot_label_tensor = torch.tensor([slot_label_id])      
        #output = (slot_label_tensor,) + outputs[1:]
        #return ((loss,) + output) if loss is not None else output

        # version 2 if return_dict is not none
        #slot_label_tensor = torch.argmax(logits.view(-1, self.num_labels), dim=1)
        #output = (slot_label_tensor,) + outputs[1:]
        #return ((loss,) + output)

        # version 2 if return_dict is none
        return TokenClassifierOutput(
            loss=loss,
            logits=torch.argmax(logits.view(-1, self.num_labels), dim=1),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        ) 


model = DistilBertForTokenClassificationFilesDomain.from_pretrained('distilbert-base-uncased', num_labels=num_labels,
                                                            output_attentions=False, output_hidden_states=False)


lr_scaler = hvd.size()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# for debug 
print("device: {}".format('cuda' if torch.cuda.is_available() else 'cpu'))

model = model.to(device)

#we print the model architecture and all model learnable parameters.
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print('Number of trainable parameters:', count_parameters(model), '\n', model)

# This code is taken from:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L102
# Don't apply weight decay to any parameters whose names include these tokens.
# (Here, the BERT doesn't have `gamma` or `beta` parameters, only `bias` terms)
no_decay = ['bias', 'LayerNorm.weight']
# Separate the `weight` parameters from the `bias` parameters. 
# - For the `weight` parameters, this specifies a 'weight_decay_rate' of 0.01. (means multiply by 0.99)
# - For the `bias` parameters, the 'weight_decay_rate' is 0.0. 
optimizer_grouped_parameters = [
    # Filter for all parameters which *don't* include 'bias', 'gamma', 'beta'.
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.2},
     # Filter for parameters which *do* include those.
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



# for debug
'''
for i in range(len(val_x)):
    for j in range(len(val_x[i])):
        print(val_x[i][j], end=' ')
    print()
'''

# https://zhuanlan.zhihu.com/p/143209797
# this link has similar process as the folloiwngf code for each function
for n in range(num_epochs):
    # Reset the total loss for this epoch.
    train_loss = 0
    val_loss = 0
    # Measure how long the training epoch takes.
    start_time = time.time()
    
    # For each batch of training data...
    for k, (mb_x, mb_m, mb_y) in enumerate(train_dataloader):

        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because 
        # accumulating the gradients is "convenient while training RNNs". 
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        optimizer.zero_grad()

        # Put the model into training mode. Don't be mislead--the call to 
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()
        
        mb_x = mb_x.to(device)
        mb_m = mb_m.to(device)
        mb_y = mb_y.to(device)
        

        # Perform a forward pass (evaluate the model on this training batch).
        # The documentation for this `model` function is here: 
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        # It returns different numbers of parameters depending on what arguments
        # arge given and what flags are set. For our useage here, it returns
        # the loss (because we provided labels) and the "logits"--the model
        # outputs prior to activation.
        outputs = model(mb_x, attention_mask=mb_m, labels=mb_y)
        
        loss = outputs[0]



        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()
        # Update the learning rate.
        scheduler.step()

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.        
        train_loss += loss.data / num_mb_train
    
    print ("\nTrain loss after itaration %i: %f" % (n+1, train_loss))
    avg_train_loss = metric_average(train_loss, 'avg_train_loss')
    print ("Average train loss after iteration %i: %f" % (n+1, avg_train_loss))
    train_losses.append(avg_train_loss)
    

    # Tell pytorch not to bother with constructing the compute graph during
    # the forward pass, since this is only needed for backprop (training).
    with torch.no_grad():
       # Put the model in evaluation mode--the dropout layers behave differently
       # during evaluation.
        model.eval()
        
        for k, (mb_x, mb_m, mb_y) in enumerate(val_dataloader):
            mb_x = mb_x.to(device)
            mb_m = mb_m.to(device)
            mb_y = mb_y.to(device)
        
            # in forward function 
            # outputs[0] is logic
            #if not return_dict:
            #output = (start_logits, end_logits) + distilbert_output[1:]
            #return ((total_loss,) + output) if total_loss is not None else output

            #return QuestionAnsweringModelOutput(
            #loss=total_loss,
            #start_logits=start_logits,
            #end_logits=end_logits,
            #hidden_states=distilbert_output.hidden_states,
            #attentions=distilbert_output.attentions,
            #)
            # Forward pass, calculate logit predictions.
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            outputs = model(mb_x, attention_mask=mb_m, labels=mb_y)
            
            loss = outputs[0]

            
            # for debug
            print ("evaluate result")
            for j in range(len(outputs[1])):
                print(outputs[1][j], end=' ')
            print()
            
            val_loss += loss.data / num_mb_val
            
        print ("Validation loss after itaration %i: %f" % (n+1, val_loss))
        avg_val_loss = metric_average(val_loss, 'avg_val_loss')
        print ("Average validation loss after iteration %i: %f" % (n+1, avg_val_loss))
        val_losses.append(avg_val_loss)
    
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f'Time: {epoch_mins}m {epoch_secs}s')

# ? not sure why needs checking rank
if hvd.rank() == 0:
    
    out_dir = './outputs'
    
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(out_dir)
    fast_tokenizer.save_pretrained(out_dir)

    with open(out_dir + '/train_losses.pkl', 'wb') as f:
        joblib.dump(train_losses, f)

    with open(out_dir + '/val_losses.pkl', 'wb') as f:
        joblib.dump(val_losses, f)


    #add to save output model in pt
    # it might be redundant since it might be the same as model_to_save.save_pretrained(out_dir)
    # but right know i use model.pt to register in azure
    torch.save(model, os.path.join(out_dir, 'model.pt'))

    run.log('validation loss', avg_val_loss)

    
    # save onnx
    # Tokenizing input text
    text = "a visually stunning rumination on love"
    fast_tokenized_batch : BatchEncoding = fast_tokenizer(text)
    fast_tokenized_text :Encoding  =fast_tokenized_batch[0]
    fast_tokenized_text_tokens_copy = fast_tokenized_text.tokens

    print("fast token ouput version 2 being used here: {}".format(fast_tokenized_text_tokens_copy))

    # Masking one of the input tokens
    # this is question answering so change it only a single sentence
    #masked_index = 8
    masked_index = 3
    fast_tokenized_text_tokens_copy[masked_index] = '[MASK]'
    indexed_tokens = fast_tokenizer.convert_tokens_to_ids(fast_tokenized_text_tokens_copy)

    # for debug 
    print("fast token ouput version 2 being used here after replace: {}".format(fast_tokenized_text_tokens_copy))

    print("indexed_tokens: {}".format(indexed_tokens))
    segments_ids = [0]

    # Creating a dummy input
    # but you need to move tensors to GPU
    #https://github.com/huggingface/transformers/issues/227
    #tokens_tensor = torch.tensor([indexed_tokens])
    #segments_tensors = torch.tensor([segments_ids])
    print("create input for device: {}".format(device))

    # adding [] is the same as unqueeze(0) function
    tokens_tensor = torch.tensor([indexed_tokens]).to(device)

    segments_tensors = torch.tensor([segments_ids]).to(device)
    dummy_input = tokens_tensor

    # for deubg
    print("tokens_tensor shape: {}".format(tokens_tensor.shape))
    print("segments_tensor shape: {}".format(segments_tensors.shape))

    print("tokens_tensor: {}".format(tokens_tensor))
    print("segments_tensor: {}".format(segments_tensors))

    

    #torch.onnx.export(model_to_save, 
    #    dummy_input, out_dir + '/traced_distill_bert.onnx', 
    #    verbose=True)


    #follow yue's suggestion to add output 
    torch.onnx.export(model=model_to_save,
        args=(dummy_input),
        f=out_dir + '/traced_distill_bert.onnx.bin',
        input_names = ["input_ids"],
        verbose=True,
        # here logits is not internal logits means
        # ? change it to a better name like slot_output in the future
        #output_names = ["logits"],
        #output_names = ["slot_label_tensor"],
        output_names = ["slot_output"],
        do_constant_folding = True,
        opset_version=11,
        #dynamic_axes = {'input_ids': {1: '?'}, 'logits': {1: '?'}}
        #dynamic_axes = {'input_ids': {1: '?'}, 'slot_label_tensor': {1: '?'}}
        dynamic_axes = {'input_ids': {1: '?'}, 'slot_output': {1: '?'}}
        )
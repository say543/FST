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
import codecs;
import string;
import traceback


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
df = pd.read_csv('E:/azure_ml_notebook/azureml_data/files_slot_training.tsv', sep='\t', encoding="utf-8")
#df = pd.read_csv('E:/azure_ml_notebook/azureml_data/files_slot_training_small.tsv', sep='\t', encoding="utf-8")
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



# read label
from typing_extensions import TypedDict
from typing import List,Any
IntList = List[int] # A list of token_ids
IntListList = List[IntList] # A List of List of token_ids, e.g. a Batch


import itertools
class LabelSet:
    def __init__(self, labels: List[str], tokenizer, useIob=False):
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


        self.cutoff_length = 100;
        self.open_pattern = r'<(\w+)>';
        self.close_pattern = r'</(\w+)>';
        self.useIob = useIob;

        self.tokenizer = tokenizer

        self.slot_list = [];

        for key, value in self.labels_to_id.items():
            self.slot_list.append('<' + key + '>');
            self.slot_list.append('</' + key + '>');

    def get_aligned_label_ids_from_aligned_label(self, aligned_labels):
        return list(map(self.labels_to_id.get, aligned_labels))



    def get_untagged_id(self):
        return self.labels_to_id["o"]

    def get_untagged_label(self):
        return self.ids_to_label[0]

    def splitWithBert(self, annotation):
        preSplit = annotation.split();
        annotationArray = [];
        for word in preSplit:
            if any(slot in word for slot in self.slot_list):
            #if any('<'+slot in word for slot in self.slolabels_to_id):
                annotationArray.append(word);
            else:
                annotationArray += self.tokenizer.tokenize(word);
        return annotationArray;

    def generateTagString(self, annotation_filtered_array):
        if self.useIob:
            preTag = '';
            annotation_filtered_array_iob = [];
            for idx, tag in enumerate(annotation_filtered_array):
                if tag == 'O':
                    annotation_filtered_array_iob.append(tag);
                    preTag = '';
                else:
                    annotation_filtered_array_iob.append(('B-' if tag != preTag else 'I-') + tag);
                    preTag = tag;

            tag_string = " ".join(annotation_filtered_array_iob);
        else:
            tag_string = " ".join(annotation_filtered_array);
        return tag_string;


    def checkQueryValid(self, input):
        # query should not contain any unwanted slot annotations (this happens when the original annotation is wrong)
        for error_item in self.slot_list:
        #for key, value in self.labels_to_id.items():

            if error_item in input:
            #if '<' + key + '>' in input:
                return False;
        return True;

    def isOpenPattern(self, word):
        match = re.match(self.open_pattern, word);
        if not match:
            return (None, False);
        slot = match[1].strip().lower();
        #if slot in self.slot_dict:
        if slot in self.labels_to_id:
            return (slot, True);
        return (None, False);

    def isClosePattern(self, word, tag):
        match = re.match(self.close_pattern, word);
        if not match:
            return False;
        slot = match[1].strip();
        return slot == tag;


    def simplePreprocessAnnotation(self, input):
        # remove email action related
        #email_action_pattern = r'<email_action>\s*([^<]+?)\s*</email_action>';
        #preprocessed = re.sub(email_action_pattern, '\\1', input);

        # remove email quantifier related
        #quantifier_pattern = r'<quantifier>\s*([^<]+?)\s*</quantifier>';
        #preprocessed = re.sub(quantifier_pattern, '\\1', preprocessed);


        preprocessed = input

        # open up the slot annotation with space
        pattern = r'<([^>]+?)>\s*([^<]+?)\s*</([^>]+?)>';
        preprocessed = re.sub(pattern, '<\\1> \\2 </\\3>', preprocessed); # <date>friday</date> -> <date> friday </date> for further splitting
        if preprocessed[-1] in string.punctuation and preprocessed[-1] != '>':
            punc = preprocessed[-1];
            preprocessed = preprocessed[:-1] + ' ' + punc;
        return preprocessed;

    def preprocessRawAnnotation(self, query, annotation_input, isTrain=True):
        pattern = r'<(?P<name>\w+)>(?P<entity>[^<]+)</(?P=name)>';

        try:
            annotation_original = annotation_input.strip().lower();
            annotation = self.simplePreprocessAnnotation(annotation_original);
            annotation_array = self.splitWithBert(annotation);
            #annotation_array = self.splitWithBert(annotation_original);
            annotation_result_arrry = [-1] * len(annotation_array);

            # capture slot name and slot entity, store them in a dict;
            # find out slot
            i = 0;
            while i < len(annotation_array):
                (slot, isOpen) = self.isOpenPattern(annotation_array[i]);
                if not isOpen:
                    annotation_result_arrry[i] = 'O';
                    i += 1;
                else:
                    j = i+1;
                    while(not self.isClosePattern(annotation_array[j], slot)):
                        annotation_result_arrry[j] = slot;
                        j += 1;
                    i = j+1;
            
            word_array = [word for idx, word in enumerate(annotation_array) if annotation_result_arrry[idx] != -1];
            annotation_filtered_array = [word for idx, word in enumerate(annotation_result_arrry) if annotation_result_arrry[idx] != -1];
            assert(len(word_array) == len(annotation_filtered_array));

            # adding cutoff length for query
            if len(word_array) == 0:
                #continue;
                return '', ''
            elif len(word_array) == 1:
                if all(i in string.punctuation for i in word_array):
                    #continue;
                    return '', ''

            if len(word_array) > self.cutoff_length:
                word_array = word_array[:self.cutoff_length];
                annotation_filtered_array = annotation_filtered_array[:self.cutoff_length];

            # write input string and tag list in file;
            word_string = " ".join(word_array);
            tag_string = self.generateTagString(annotation_filtered_array);

            if not self.checkQueryValid(word_string):
                #continue;
                return '', ''

            #feature_parse.append((word_string, tag_string));
            return word_string, tag_string
        except:
            # print stack traice
            traceback.print_exc()
            print("skipped query: {} and pair: {}".format(query, annotation_input))
            return '', '' 
            #continue;

slots = ["o", 
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
slots_label_set = LabelSet(labels=map(str.lower,slots), 
                            tokenizer =fast_tokenizer)




# iterative get labele and also append padding based on text_ids
tokensForQueries = []
labelsForQueries =[]

for i, row in df.iterrows():
    
    '''
	line = line.strip();
	if not line:
		continue;
	linestrs = line.split("\t");
	# make sure it at least has
	# Query	ExternalFeature	Weight	Intent	Domain	Slot
	if len(linestrs) < 5:
		continue;
    '''

    query = row['query']
    slot = row['QueryXml']
	# remove head and end spaces 
    slot = slot.strip()


    # for debug 
    #print("query:{}".format(query))
    #print("slot: {}".format(slot))

    annotations = []


    word_string, tag_string  = slots_label_set.preprocessRawAnnotation(query, slot)
    if word_string == '' and tag_string == '':
        continue

    tokensForQueries.append(word_string)
    labelsForQueries.append(tag_string)

    '''
    # for contact_name to reanme to to_contact_name
    xmlpairs = re.findall("(<.*?>.*?<\/.*?>)", slot)

    queryIndex = 0
    for xmlpair in xmlpairs:
        # extra type and value for xml tag
        xmlTypeEndInd = xmlpair.find(">")

        xmlType = xmlpair[1:xmlTypeEndInd]

        xmlValue = xmlpair.replace("<"+xmlType+">", "")
        xmlValue = xmlValue.replace("</"+xmlType+">", "")
        xmlValue = xmlValue.strip()

        start = query.lower()[queryIndex:].find(xmlValue.lower())
        if start == -1:
            print("skipped query: {} and pair: {}".format(query, xmlValue))
            continue

        # record annotation index
        annotations.append(dict(start=queryIndex+start,end=queryIndex+start+len(xmlValue),text=xmlValue,label=xmlType))

        # update queryIndex according to moving order
        queryIndex = queryIndex+start+len(xmlValue)

        
    # for debug
    #for anno in annotations:
    ##Show our annotations
	#    print (query[anno['start']:anno['end']],anno['label'])

    fast_tokenized_batch : BatchEncoding = fast_tokenizer(query)
    fast_tokenized_text :Encoding  =fast_tokenized_batch[0]

    # fast token will add CLS and SEP 
    # ? not sure in real trainnig , do we need to provide or not
    # in yue case it does not include those two
    print("fast token ouput: {}".format(fast_tokenized_text.tokens))

    tokens = fast_tokenized_text.tokens
    aligned_labels = ["O"]*len(tokens) # Make a list to store our labels the same length as our tokens
    for anno in (annotations):
        for char_ix in range(anno['start'],anno['end']):
            token_ix = fast_tokenized_text.char_to_token(char_ix)
            if token_ix is not None: # White spaces have no token and will return None
                aligned_labels[token_ix] = anno['label']
    '''

    # for debug
    #for token,label in zip(tokens,aligned_labels):
    #    print (token,"-",label) 


    # for collect query and label
    #query = ''
    ##labelsforQuery = ''
    #for i, (token, label) in enumerate(zip(tokens, aligned_labels)):
    #    # ignore [CLS] / [SEP]
    #    if i == 0 or i == len(tokens)-1:
    #        continue
    #    query = query + ' ' + token
    #    labelsforQuery = labelsforQuery + ' '+ str(label)

    #tokensForQueries.append(query)
    #labelsForQueries.append(labelsforQuery)

    #aligned_label_ids = slots_label_set.get_aligned_label_ids_from_aligned_label(
    #    map(str.lower,aligned_labels)
    #)

    # for debug
    #for token, label in zip(tokens, aligned_label_ids):
    #    print(token, "-", label)



print('output training and test file');
#output_file = 'train_bert_email_slot.tsv' if isTrain == True else 'dev_bert_email_slot.tsv'
output_file = '..\\azureml_data\\files_slot_train_after_my_preprocessing.tsv'
with codecs.open(output_file, 'w', 'utf-8') as fout:
    for i, (query, labelsforQuery) in enumerate(zip(tokensForQueries, labelsForQueries)):
        fout.write(query.strip() + '\t' + labelsforQuery.strip() + '\r\n');



'''
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
'''

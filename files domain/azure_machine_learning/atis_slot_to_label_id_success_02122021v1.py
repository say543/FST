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

df = pd.read_csv('E:/azure_ml_notebook/azureml_data/atis_train.tsv', sep='\t', encoding="utf-8",
    keep_default_na=False,
    dtype={
    'MessageId': object, 'Frequency': object, 'ConversationContext': object, 'SelectionIgnore': object})

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

    def get_id(self, label):
        return self.labels_to_id[label]

    def get_label(self, id):
        return self.ids_to_label[id]        


    
    def get_ids(self):
        return self.labels_to_id

    def get_labels(self):
        return self.labels_to_id


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


    def preprocessIOBAnnotation(self, query, annotation_input, isTrain=True):
        new_annotation_input = ""
        #  convert_examples_to_features
        # _create_examples
        try:
            query_original = query.strip().lower();
            query_original_words = query_original.split()
            annotation_original = annotation_input.strip().lower();
            annotation_original_labels =  annotation_original.split()
        
            assert len(query_original_words) == len(annotation_original_labels)


            # ? not sure how this value works
            pad_token_label_id = -100
            tokens = []
            slot_labels = []
            for word, label in zip(query_original_words, annotation_original_labels):
                word_tokens = self.tokenizer.tokenize(word)
                # ? not sure if this needed
                if not word_tokens: 
                    raise Exception(query, 'unsopported tokens in word')
                    #word_t word_tokens:
                    #word_tokens = [unk_token]  # For handling the bad-encoded word

                tokens.extend(word_tokens)
                # do not use pad_token_label_id from ignore_index
                #slot_labels_ids.extend([int(slot_label)] + [pad_token_label_id] * (len(word_tokens) - 1))
                for i in range(len(word_tokens)):
                    slot_labels.append(label)


            return ' '.join([str(token) for token in tokens]), ' '.join([str(slot_label) for slot_label in slot_labels])
        except:
            # print stack traice
            traceback.print_exc()
            print("<IOB> skipped query: {} and pair: {}".format(query, annotation_input))
            return '', '' 
            #continue;

    def preprocessRawAnnotation(self, query, annotation_input, isTrain=True, useIob=False):

        if useIob is True:
            return self.preprocessIOBAnnotation(query, annotation_input, isTrain=True)

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

slots = [
    "o",
    "B-aircraft_code",
    "B-airline_code",
    "B-airline_name",
    "I-airline_name",
    "B-airport_code",
    "B-airport_name",
    "I-airport_name",
    "B-arrive_date.date_relative",
    "B-arrive_date.day_name",
    "B-arrive_date.day_number",
    "I-arrive_date.day_number",
    "B-arrive_date.month_name",
    "B-arrive_date.today_relative",
    "B-arrive_time.end_time",
    "I-arrive_time.end_time",
    "B-arrive_time.period_mod",
    "B-arrive_time.period_of_day",
    "I-arrive_time.period_of_day",
    "B-arrive_time.start_time",
    "I-arrive_time.start_time",
    "B-arrive_time.time",
    "I-arrive_time.time",
    "B-arrive_time.time_relative",
    "I-arrive_time.time_relative",
    "B-city_name",
    "I-city_name",
    "B-class_type",
    "I-class_type",
    "B-connect",
    "B-cost_relative",
    "I-cost_relative",
    "B-day_name",
    "B-day_number",
    "B-days_code",
    "B-depart_date.date_relative",
    "B-depart_date.day_name",
    "B-depart_date.day_number",
    "I-depart_date.day_number",
    "B-depart_date.month_name",
    "B-depart_date.today_relative",
    "I-depart_date.today_relative",
    "B-depart_date.year",
    "B-depart_time.end_time",
    "I-depart_time.end_time",
    "B-depart_time.period_mod",
    "B-depart_time.period_of_day",
    "I-depart_time.period_of_day",
    "B-depart_time.start_time",
    "I-depart_time.start_time",
    "B-depart_time.time",
    "I-depart_time.time",
    "B-depart_time.time_relative",
    "I-depart_time.time_relative",
    "B-economy",
    "I-economy",
    "B-fare_amount",
    "I-fare_amount",
    "B-fare_basis_code",
    "I-fare_basis_code",
    "B-flight_days",
    "B-flight_mod",
    "I-flight_mod",
    "B-flight_number",
    "B-flight_stop",
    "I-flight_stop",
    "B-flight_time",
    "I-flight_time",
    "B-fromloc.airport_code",
    "B-fromloc.airport_name",
    "I-fromloc.airport_name",
    "B-fromloc.city_name",
    "I-fromloc.city_name",
    "B-fromloc.state_code",
    "B-fromloc.state_name",
    "I-fromloc.state_name",
    "B-meal",
    "B-meal_code",
    "I-meal_code",
    "B-meal_description",
    "I-meal_description",
    "B-mod",
    "B-month_name",
    "B-or",
    "B-period_of_day",
    "B-restriction_code",
    "I-restriction_code",
    "B-return_date.date_relative",
    "I-return_date.date_relative",
    "B-return_date.day_name",
    "B-return_date.day_number",
    "B-return_date.month_name",
    "B-return_date.today_relative",
    "I-return_date.today_relative",
    "B-return_time.period_mod",
    "B-return_time.period_of_day",
    "B-round_trip",
    "I-round_trip",
    "B-state_code",
    "B-state_name",
    "B-stoploc.airport_name",
    "B-stoploc.city_name",
    "I-stoploc.city_name",
    "B-stoploc.state_code",
    "B-time",
    "I-time",
    "B-time_relative",
    "B-today_relative",
    "I-today_relative",
    "B-toloc.airport_code",
    "B-toloc.airport_name",
    "I-toloc.airport_name",
    "B-toloc.city_name",
    "I-toloc.city_name",
    "B-toloc.country_name",
    "B-toloc.state_code",
    "B-toloc.state_name",
    "I-toloc.state_name",
    "B-transport_type",
    "I-transport_type"
]


# map all slots to lower case
slots_label_set = LabelSet(labels=map(str.lower,slots), 
                            tokenizer =fast_tokenizer)

class IntentLabelSet:
    def __init__(self, labels: List[str]):
        self.labels_to_id = {}
        self.ids_to_label = {}

        num = 0
        for label in labels:
            self.labels_to_id[label] = num
            self.ids_to_label[num] = label
            num = num +1 

    def get_labels(self):
        return self.labels_to_id

    def get_id(self, label):
        return self.labels_to_id[label]

    def get_label(self, id):
        return self.ids_to_label[id]  
    
    def get_ids_from_label(self, label):
        return self.labels_to_id[label]


# ? multi turn intent how to incorporate extra features is not yet decided
intents = [
    "x",
    "atis_abbreviation",
    "atis_aircraft",
    "atis_aircraft#atis_flight",
    "atis_airfare",
    "atis_airline",
    "atis_airline#atis_flight_no",
    "atis_airport",
    "atis_capacity",
    "atis_cheapest",
    "atis_city",
    "atis_distance",
    "atis_flight",
    "atis_flight#atis_airfare",
    "atis_flight_no",
    "atis_flight_time",
    "atis_ground_fare",
    "atis_ground_service",
    "atis_ground_service#atis_ground_fare",
    "atis_meal",
    "atis_quantity",
    "atis_restriction"
]


intent_label_set = IntentLabelSet(labels=map(str.lower,intents))



# not yet finished
'''
class EvaluationFromYue():
    def __init__(self, slots_label_set, intent_label_set, useIob=False):
        self.useIob = useIob;
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu");
        # no need
        #self.preprocess_driver = Preprocess_Driver_Bert(self.useIob);
        #self.slot_dict = self.preprocess_driver.slot_dict;
        #self.slot_dict_rev = {v: k for k, v in self.slot_dict.items()};

        # replace with my own class 
        #self.intent_dict = self.preprocess_driver.intent_dict;
        #self.intent_dict_rev = {v: k for k, v in self.intent_dict.items()};
        self.slots_label_set = slots_label_set
        self.intent_label_set = intent_label_set


        # no need onnx model snice only evaluation
        #self.onnx_model_name = 'model.onnx.bin';

    def create_slot_arrays_iob(input_tag_list):
        slot_arrays={}
        for label in get_slot_labels():
            slot_arrays[label]=[]

        current_slot_array = []
        current_slot_tag = ''
        for index, word in enumerate(input_tag_list):
            if word[0:2] == 'B-':
                if current_slot_tag != '':
                    slot_arrays[current_slot_tag].append(current_slot_array)
                    current_slot_tag=''
                    current_slot_array=[]
                current_slot_tag = word[2:len(word)]
                current_slot_array.append(index)
            if word[0:2] == 'I-' and current_slot_tag != '':
                if current_slot_tag != word[2:len(word)]:
                    slot_arrays[current_slot_tag].append(current_slot_array)
                    current_slot_tag=''
                    current_slot_array=[]
                else:
                    current_slot_array.append(index)
            if (word == 'O' or index == len(input_tag_list)-1) and current_slot_tag != '':
                slot_arrays[current_slot_tag].append(current_slot_array)
                current_slot_tag=''
                current_slot_array=[]
            if word == 'O':
                continue;
        return slot_arrays;

    def create_slot_arrays(self, input_tag_list):
        slot_arrays={}
        for label in self.get_slot_labels():
            slot_arrays[label]=[]

        current_slot_array = []
        current_slot_tag = ''
        for index, word in enumerate(input_tag_list):
            if word != 'O':
                if current_slot_tag != '':
                    if current_slot_tag == word:
                        current_slot_array.append(index)
                    else:
                        slot_arrays[current_slot_tag].append(current_slot_array)
                        current_slot_array=[]
                        current_slot_tag=word
                else:
                    current_slot_tag = word
                    current_slot_array = []
                    current_slot_array.append(index)
            if (word == 'O' or index == len(input_tag_list)-1) and current_slot_tag != '':
                slot_arrays[current_slot_tag].append(current_slot_array)
                current_slot_tag=''
                current_slot_array=[]
            if word == 'O':
                continue;
        return slot_arrays;

    def get_slot_labels(self):
        slot_labels=[]
        for label in self.slot_dict.keys():
            if label != 'O':
                slot_labels.append(label)
        return slot_labels;


    def get_lexicon_vector(self, word_list):
        lexicon_list = [];
        for word in word_list:
            lexicon_list.append(self.preprocess_driver.lexiconLookup(word));
        return lexicon_list;

    def evaluate_model(self, 
                       model, 
                       input_file_tsv, 
                       jointIntent=False):

        model.eval();
        slot_labels = self.get_slot_labels()
        slot_tp_tn_fn_counts = {}

        for label in slot_labels:
            slot_tp_tn_fn_counts[label+"_fn"]=0
            slot_tp_tn_fn_counts[label+"_fp"]=0
            slot_tp_tn_fn_counts[label+"_tp"]=0

        output=""

        total_number = 0;
        total_true_positive = 0;
        total_false_positive = 0;
        total_true_negative = 0;
        total_false_negative = 0;

        intents_tp = {};
        intents_fp = {};
        intents_fn = {};

        for key in self.intent_dict:
            intents_tp[key] = 0;
            intents_fp[key] = 0;
            intents_fn[key] = 0;


        with codecs.open(input_file_tsv, 'r', 'utf-8') as f_input:
            for line in f_input:
                line = line.strip();
                if not line:
                    continue;
                array = line.split('\t');
                if len(array) < 2:
                    continue;
                sentence = array[0].strip();
                tagString = array[1].strip();
                if not sentence:
                    print('input cannot be empty');
                    continue;

                input_tag_list=tagString.split(' ');

                golden_intent = "";
                if jointIntent:
                    golden_intent = input_tag_list[0];
                    if golden_intent not in self.intent_dict:
                        continue;

                    input_tag_list = input_tag_list[1:];

                input_tag_list.append('O');
                input_tag_list.insert(0, 'O');
                golden_slot_arrays = self.create_slot_arrays_iob(input_tag_list) if self.useIob else self.create_slot_arrays(input_tag_list);
                
                word_list = sentence.split(' ');
                encoded_dict = self.preprocess_driver.tokenizer.encode_plus(word_list);

                index_array = encoded_dict['input_ids'];           
                attention_mask = encoded_dict['attention_mask'];

                assert(len(input_tag_list) == len(index_array));

                length = len(index_array);
                lexicon_index_array = [];

                with torch.no_grad():
                    input_x = torch.LongTensor(index_array).unsqueeze(0).to(self.device);
                    mask = torch.LongTensor(attention_mask).unsqueeze(0).to(self.device);
                    intent_out, slot_out = model(input_ids=input_x, attention_mask=mask);

                    intent_result = intent_out[0].cpu().tolist();
                    slot_result = slot_out[0].cpu().tolist();

                # Slot Evaluation

                predicted_labels_slot = list(map((lambda x: self.slot_dict_rev[x]), slot_result));
                predicted_slot_arrays = self.create_slot_arrays_iob(predicted_labels_slot) if self.useIob else self.create_slot_arrays(predicted_labels_slot);
                query_fn = 0
                query_fp = 0
                for label in slot_labels:
                    golden_set=set(map(tuple, golden_slot_arrays[label]))
                    prediction_set=set(map(tuple, predicted_slot_arrays[label]))
                    tp_count = len(prediction_set & golden_set)
                    fp_count = len(prediction_set-golden_set)
                    fn_count = len(golden_set-prediction_set)
                    slot_tp_tn_fn_counts[label+"_fn"]=slot_tp_tn_fn_counts[label+"_fn"]+fn_count
                    slot_tp_tn_fn_counts[label+"_fp"]=slot_tp_tn_fn_counts[label+"_fp"]+fp_count
                    slot_tp_tn_fn_counts[label+"_tp"]=slot_tp_tn_fn_counts[label+"_tp"]+tp_count
                    query_fn = query_fn+fn_count
                    query_fp = query_fp+fp_count
                
                # Intent Evaluation
                if jointIntent:
                    predicted_label_intent = self.intent_dict_rev[intent_result];

                    if predicted_label_intent == golden_intent:
                        intents_tp[golden_intent] += 1;
                    else:
                        intents_fp[predicted_label_intent] += 1;
                        intents_fn[golden_intent] += 1;

            total_tp_slot = 0;
            total_fp_slot = 0;
            total_fn_slot = 0;

            for label in slot_labels:
                total_tp_slot += slot_tp_tn_fn_counts[label+"_tp"];
                total_fp_slot += slot_tp_tn_fn_counts[label+"_fp"];
                total_fn_slot += slot_tp_tn_fn_counts[label+"_fn"];
                
                slot_metric = label+'\t'+ ': total_tp: '+str(slot_tp_tn_fn_counts[label+"_tp"])+', total_fp: '+str(slot_tp_tn_fn_counts[label+"_fp"])+', total_fn: '+str(slot_tp_tn_fn_counts[label+"_fn"]);
                print(slot_metric+'\n')
            
            overall_precision_slot = 0;
            overall_recall_slot = 0;

            if total_tp_slot != 0:
                overall_precision_slot = total_tp_slot / (total_tp_slot + total_fp_slot);
                overall_recall_slot = total_tp_slot / (total_tp_slot + total_fn_slot);

            print("overall slot precision: {}, overall slot recall: {}".format(overall_precision_slot, overall_recall_slot));

            if jointIntent:

                total_tp_intent = 0;
                total_fp_intent = 0;
                total_fn_intent = 0;

                for key in self.intent_dict:
                    total_tp_intent += intents_tp[key];
                    total_fp_intent += intents_fp[key];
                    total_fn_intent += intents_fn[key];
            
                precision_intent = 0;
                recall_intent = 0;
                if intents_tp[key] != 0:
                    precision = intents_tp[key] / (intents_tp[key] + intents_fp[key]);
                    recall = intents_tp[key] / (intents_tp[key] + intents_fn[key]);
            
                    # print("intent: {}, precision: {}, recall: {}\n".format(key, precision, recall));

                overall_precision_intent = 0;
                overall_recall_intent = 0;
                if total_tp_intent != 0:
                    overall_precision_intent = total_tp_intent / (total_tp_intent + total_fp_intent);
                    overall_recall_intent = total_tp_intent / (total_tp_intent + total_fn_intent);

                print("overall intent precision: {}, overall intent recall: {}".format(overall_precision_intent, overall_recall_intent));

    def evaluate_tsv(self, 
                     model_name, 
                     input_file_tsv, 
                     ignored_queries_tsv, 
                     output_file_tsv, 
                     slot_metrics_tsv,
                     joinIntent=False):
        
        model = Bert_Model_Slot.from_pretrained(model_name);
        model.eval();
        slot_labels = self.get_slot_labels()

        slot_tp_tn_fn_counts = {}

        for label in slot_labels:
            slot_tp_tn_fn_counts[label+"_fn"]=0
            slot_tp_tn_fn_counts[label+"_fp"]=0
            slot_tp_tn_fn_counts[label+"_tp"]=0

        intents_tp = {};
        intents_fp = {};
        intents_fn = {};

        for key in self.intent_dict:
            intents_tp[key] = 0;
            intents_fp[key] = 0;
            intents_fn[key] = 0;
        
        f_ignored_queries = open(ignored_queries_tsv, "w")
        f_output = open(output_file_tsv, "w")
        output=""

        total_number = 0;
        total_true_positive = 0;
        total_false_positive = 0;
        total_true_negative = 0;
        total_false_negative = 0;

        line_count=0
        ignore_line_count=0
        header=''
        ignore_header=''

        with codecs.open(input_file_tsv, 'r', 'utf-8') as f_input:
            for line in f_input:
                line = line.strip();
                if not line:
                    continue;
                array = line.split('\t');
                if len(array) < 2:
                    continue;
                sentence = array[0].strip();
                tagString = array[1].strip();
                if not sentence:
                    print('input cannot be empty');
                    continue;

                input_tag_list=tagString.split(' ');
                input_tag_list.append('O');
                input_tag_list.insert(0, 'O');
                golden_slot_arrays = self.create_slot_arrays(input_tag_list)

                word_list = sentence.split(' ');
                index_array = self.preprocess_driver.tokenizer.encode_plus(word_list)['input_ids'];
                assert(len(input_tag_list) == len(index_array));
                length = len(index_array);

                lexicon_index_array = [];

                with torch.no_grad():
                    input_x = torch.LongTensor(index_array).unsqueeze(0).cpu();
                    model_out = model(input_ids=input_x);
                    
                    result = model_out[0].cpu().tolist();

                predicted_labels = list(map((lambda x: self.slot_dict_rev[x]), result));

                if (len(input_tag_list) != len(index_array) or len(input_tag_list) != len(predicted_labels)):
                    if(ignore_line_count==0):
                        ignore_header="InputQuery"+'\t'+"InputLabels"+'\t'+"OutputLabels"+'\t'+"InputLabelQueryLengthMismatch"+'\t'+"InputOutputLabelLengthMismatch"
                        f_ignored_queries.write(ignore_header +'\n')
                    f_ignored_queries.write(line+'\t'+" ".join(predicted_labels)+'\t'+str(len(input_tag_list) != len(word_list))+'\t'+str(len(input_tag_list) != len(predicted_labels))+'\n')
                    ignore_line_count = ignore_line_count+1
                    continue;

                predicted_slot_arrays = self.create_slot_arrays(predicted_labels)

                if(line_count==0):
                    header="Query"+'\t'+"InputLabels"+'\t'+"PredictedLabels"
                output = sentence+'\t'+tagString+'\t'+" ".join(predicted_labels)

                query_fn = 0
                query_fp = 0
                for label in slot_labels:
                    golden_set=set(map(tuple, golden_slot_arrays[label]))
                    prediction_set=set(map(tuple, predicted_slot_arrays[label]))
                    tp_count = len(prediction_set & golden_set)
                    fp_count = len(prediction_set-golden_set)
                    fn_count = len(golden_set-prediction_set)
                    if(line_count==0):
                        header=header+'\t'+"{}_tp".format(label)+'\t'+"{}_fp".format(label)+'\t'+"{}_fn".format(label)
                    output=output+'\t'+str(tp_count)+'\t'+str(fp_count)+'\t'+str(fn_count)
                    slot_tp_tn_fn_counts[label+"_fn"]=slot_tp_tn_fn_counts[label+"_fn"]+fn_count
                    slot_tp_tn_fn_counts[label+"_fp"]=slot_tp_tn_fn_counts[label+"_fp"]+fp_count
                    slot_tp_tn_fn_counts[label+"_tp"]=slot_tp_tn_fn_counts[label+"_tp"]+tp_count
                    query_fn = query_fn+fn_count
                    query_fp = query_fp+fp_count

                if(line_count==0):
                    header=header+'\t'+"AreAllPredictionsCorrect"
                output=output+'\t'+str(query_fn==0 and query_fp==0)

                if(line_count==0):
                    f_output.write(header+'\n')
                f_output.write(output+'\n')
                line_count=line_count+1

            f_slot_metrics = open(slot_metrics_tsv, "w")

            total_tp = 0;
            total_fp = 0;
            total_fn = 0;

            for label in slot_labels:
                slot_precision = 0;
                slot_recall = 0;
                if slot_tp_tn_fn_counts[label+"_tp"] != 0:
                    slot_precision = slot_tp_tn_fn_counts[label+"_tp"]/(slot_tp_tn_fn_counts[label+"_tp"]+slot_tp_tn_fn_counts[label+"_fp"])
                    slot_recall = slot_tp_tn_fn_counts[label+"_tp"]/(slot_tp_tn_fn_counts[label+"_tp"]+slot_tp_tn_fn_counts[label+"_fn"])
                slot_metric = label+'\t'+ ': total_tp: '+str(slot_tp_tn_fn_counts[label+"_tp"])+', total_fp: '+str(slot_tp_tn_fn_counts[label+"_fp"])+', total_fn: '+str(slot_tp_tn_fn_counts[label+"_fn"])+', precision: '+str(slot_precision)+'\t'+', recall: '+str(slot_recall)
                f_slot_metrics.write(slot_metric+'\n')
                print(slot_metric+'\n')

                total_tp += slot_tp_tn_fn_counts[label+"_tp"];
                total_fp += slot_tp_tn_fn_counts[label+"_fp"];
                total_fn += slot_tp_tn_fn_counts[label+"_fn"];
            
            overall_precision = 0;
            overall_recall = 0;
            if total_tp != 0:
                overall_precision = total_tp / (total_tp + total_fp);
                overall_recall = total_tp / (total_tp + total_fn);
            
            print('overall precision: ' + str(overall_precision) + '\r\n');
            print('overall recall: ' + str(overall_recall) + '\r\n');

            f_slot_metrics.close()
            f_ignored_queries.close()
            f_output.close()

    def evaluate_model_intent(self, input_file_tsv, model=None, model_location=None, threshold=None):
        
        if model is None:
            if model_location is None:
                raise("Must provide model location");
            model = Bert_Model_Intent.from_pretrained(model_location);
        
        model.eval();
        intents_tp = {};
        intents_fp = {};
        intents_fn = {};

        for key in self.intent_dict:
            intents_tp[key] = 0;
            intents_fp[key] = 0;
            intents_fn[key] = 0;

        with codecs.open(input_file_tsv, 'r', 'utf-8') as f_input:
            for line in f_input:
                line = line.strip();
                if not line:
                    continue;
                array = line.split('\t');
                if len(array) < 2:
                    continue;
                sentence = array[0].strip();
                intents = array[1].strip();
                if not sentence:
                    print('input cannot be empty');
                    continue;

                word_list = sentence.split(' ');
                golden_intent_list = [];

                # Evaluate Multiple Intent
                if threshold is not None:
                    intent_list_raw = intents.split(',');
                    golden_intent_list = [item.strip() for item in intent_list_raw];
                else:
                    golden_intent_list.append(intents);
                
                # If intent is not in dict, do not evaluate
                intent_valid = True;
                for item in golden_intent_list:
                    if item not in self.intent_dict:
                        intent_valid = False;
                        break;

                if not intent_valid:
                    continue;

                index_array = self.preprocess_driver.tokenizer.encode_plus(word_list)['input_ids'];

                with torch.no_grad():
                    input_x = torch.LongTensor(index_array).unsqueeze(0).to(self.device);
                    model_out = model(input_ids=input_x);

                    result = model_out[0].cpu().tolist();
                
                predicted_intent_list = [];

                # Evaluate Multiple intent where the predicted score is beyond a threshold
                if threshold is not None:
                    for idx in range(len(result)):
                        if result[idx] > threshold:
                            predicted_intent_list.append(self.intent_dict_rev[idx]);
                else:
                    predicted_intent_list.append(self.intent_dict_rev[result]);

                for item in golden_intent_list:
                    if item in predicted_intent_list:
                        intents_tp[item] += 1;
                    else:
                        intents_fn[item] += 1;

                for item in predicted_intent_list:
                    if item not in golden_intent_list:
                        intents_fp[item] += 1;

        total_tp = 0;
        total_fp = 0;
        total_fn = 0;

        for key in self.intent_dict:
            total_tp += intents_tp[key];
            total_fp += intents_fp[key];
            total_fn += intents_fn[key];
            
            precision = 0;
            recall = 0;
            if intents_tp[key] != 0:
                precision = intents_tp[key] / (intents_tp[key] + intents_fp[key]);
                recall = intents_tp[key] / (intents_tp[key] + intents_fn[key]);
            
            print("intent: {}, precision: {}, recall: {}\n".format(key, precision, recall));

        overall_precision = 0;
        overall_recall = 0;
        if total_tp != 0:
            overall_precision = total_tp / (total_tp + total_fp);
            overall_recall = total_tp / (total_tp + total_fn);

        print("overall precision: {}, overall recall: {}".format(overall_precision, overall_recall));
                
    def saveModelOnnx(self, model_name):
        model = Bert_Model_Slot.from_pretrained(model_name);
        model.eval();
        print(model);
        pytorch_total_params = sum(p.numel() for p in model.parameters());
        print(pytorch_total_params);

        print('save onnx');
        dummy_inputs = [101, 3191, 2026, 22028, 2055, 2522, 17258, 10651, 102];
        dummy_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1];
        
        inputs = torch.LongTensor(dummy_inputs).unsqueeze(0).cpu();
        masks = torch.LongTensor(dummy_mask).unsqueeze(0).cpu();
        torch.onnx.export(model=model, 
                          args=(inputs, masks),
                          f=self.onnx_model_name,
                          input_names = ["input_ids", "attention_mask"],
                          verbose=True,
                          output_names = ["slot_output"],
                          do_constant_folding = True,
                          opset_version=11,
                          dynamic_axes = {'input_ids': {1: '?'}, 'attention_mask': {1: '?'}, 'slot_output': {1: '?'}});
'''

class Evaluation():
    def __init__(self, slots_label_set, intent_label_set, useIob=False):
        self.useIob = useIob;
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu");
        # no need
        #self.preprocess_driver = Preprocess_Driver_Bert(self.useIob);
        #self.slot_dict = self.preprocess_driver.slot_dict;
        #self.slot_dict_rev = {v: k for k, v in self.slot_dict.items()};

        # replace with my own class 
        #self.intent_dict = self.preprocess_driver.intent_dict;
        #self.intent_dict_rev = {v: k for k, v in self.intent_dict.items()};
        self.slots_label_set = slots_label_set
        self.intent_label_set = intent_label_set


        #self.intent_preds = []
        #self.out_intent_label_ids =[] 
        #self.slot_preds = []
        #self.out_slot_labels_ids = []

        self.intent_preds =  None
        #self.out_intent_label = None
        self.slot_preds = None
        #self.out_slot_labels = None


        self.intent_golden = None
        self.slot_golden = None

    def add_intent_pred_and_golden(self, intent_output, intent_golden):
        #intent_preds = np.append(intent_preds, intent_logits.detach().cpu().numpy(), axis=0)
        # single dimention
        #intent_preds = np.append(intent_preds, intent_label_set.get_label(intent_logits.item()), axis=0)
        #out_intent_label_ids = np.append(
        #     out_intent_label_ids, intent_logits.item(), axis=0)

        #self.intent_preds.append(intent_label_set.get_label(intent_output.item()))
        #self.out_intent_label_ids.append(intent_output.item())

        # intent_output is a two 2D tensor
        
        if self.intent_preds is None:
            intent_list = intent_output.tolist()
            self.intent_preds = intent_list

            intent_golden_list = intent_golden.tolist()
            self.intent_golden = intent_golden_list           
            #self.out_intent_label = [intent_label_set.get_label(intent) for intent in intent_list]

            # change to numpy
            self.intent_preds = np.array(self.intent_preds)
            self.intent_golden = np.array(self.intent_golden)

            #self.out_intent_label = np.array(self.out_intent_label)
        else:
            intent_list = intent_output.tolist()
            self.intent_preds = np.append(self.intent_preds, intent_list, axis=0)

            intent_golden_list = intent_golden.tolist()
            self.intent_golden = np.append(self.intent_golden, intent_golden_list, axis=0)

            #self.out_intent_label = np.append(self.out_intent_label, [intent_label_set.get_label(intent) for intent in intent_list], axis=0)

    def add_slot_pred_and_golden(self, slot_output, slot_golden):
        #intent_preds = np.append(intent_preds, intent_logits.detach().cpu().numpy(), axis=0)
        # single dimention
        #intent_preds = np.append(intent_preds, intent_label_set.get_label(intent_logits.item()), axis=0)
        #out_intent_label_ids = np.append(
        #     out_intent_label_ids, intent_logits.item(), axis=0)


        #slot_list = []
        #for ele in slot_output.tolist():
        #    slot_list.append(slots_label_set.get_label(ele))
        #self.slot_preds.append(slot_list)
        
        #slot_id_list = []
        #for ele in slot_output.tolist():
        #    slot_id_list.append(ele)
        #self.out_slot_labels_ids.append(slot_id_list)

        if self.slot_preds is None:
            slot_id_list = slot_output.tolist()
            self.slot_preds = slot_id_list

            slot_golden_id_list = slot_golden.tolist()
            self.slot_golden = slot_golden_id_list            

            #slot_labels_list = []
            #for i in range(len(slot_id_list)):
            #    sub_list =  [slots_label_set.get_label(slot_id) for slot_id in slot_id_list[i]]
            #    slot_labels_list.append(sub_list)

            # change to numpy
            self.slot_preds = np.array(self.slot_preds)
            self.slot_golden = np.array(self.slot_golden)
            #self.out_slot_labels = np.array(slot_labels_list)

        else:

            slot_id_list = slot_output.tolist()
            self.slot_preds = np.append(self.slot_preds, slot_id_list, axis=0)

            slot_golden_id_list = slot_golden.tolist()
            self.slot_golden = np.append(self.slot_golden, slot_golden_id_list, axis=0)

            #slot_labels_list = []
            #for i in range(len(slot_id_list)):
            #    sub_list =  [slots_label_set.get_label(slot_id) for slot_id in slot_id_list[i]]
            #    slot_labels_list.append(sub_list)

            #self.out_slot_labels = np.append(self.out_slot_labels, slot_labels_list, axis=0)


    def get_intent_metrics(self, preds, golden):
        #acc = (preds == golden).mean()
        #return {
        #    "intent_acc": acc
        #}
        assert len(preds) == len(golden)


        intents_tp = {};
        intents_fp = {};
        intents_fn = {};

        for label in self.intent_label_set.get_labels():
            intents_tp[label] = 0;
            intents_fp[label] = 0;
            intents_fn[label] = 0;

        for pred, golden_per_query in zip(preds.tolist(), golden.tolist()):
            pred_label = self.intent_label_set.get_label(pred)
            golden_per_query_label = self.intent_label_set.get_label(golden_per_query)
            if pred_label == golden_per_query_label:
                intents_tp[golden_per_query_label] += 1;
            else:
                intents_fp[pred_label] += 1;
                intents_fn[golden_per_query_label] += 1;                

        total_tp_intent = 0;
        total_fp_intent = 0;
        total_fn_intent = 0;

        for label in self.intent_label_set.get_labels():
            total_tp_intent += intents_tp[label];
            total_fp_intent += intents_fp[label];
            total_fn_intent += intents_fn[label];

            if intents_tp[label] != 0:
                precision = intents_tp[label] / (intents_tp[label] + intents_fp[label]);
                recall = intents_tp[label] / (intents_tp[label] + intents_fn[label]);
            
                # for each label metric
                print("intent: {}, precision: {}, recall: {}\n".format(label, precision, recall));
        
        # for debug
        overall_precision_intent = 0;
        overall_recall_intent = 0;
        if total_tp_intent != 0:
            overall_precision_intent = total_tp_intent / (total_tp_intent + total_fp_intent);
            overall_recall_intent = total_tp_intent / (total_tp_intent + total_fn_intent);

        print("overall intent precision: {}, overall intent recall: {}".format(overall_precision_intent, overall_recall_intent));
        
        return {
            "total_intent_precision": overall_precision_intent,
            "total_intent_recall": overall_recall_intent
        }

    # for iob
    #def get_intent_metrics(self, preds, golden):
    #    #acc = (preds == golden).mean()
    #    #return {
    #    #    "intent_acc": acc
    #    #}
    #    assert len(preds) == len(golden)
       
    #    return {
    #        "intent_precision": precision_score(preds, golden),
    #        "intent_recall": recall_score(preds, golden)
    #    }


    def create_slot_arrays(self, pred_list):
        # initailize each slot's result
        slot_arrays={}
        for label in self.slots_label_set.get_labels():
            slot_arrays[label]=[]

        current_slot_array = []
        current_slot_tag = ''
        for index, id in enumerate(pred_list):
            word = self.slots_label_set.get_label(id)
            if word != 'o':
                if current_slot_tag != '':
                    if current_slot_tag == word:
                        current_slot_array.append(index)
                    else:
                        slot_arrays[current_slot_tag].append(current_slot_array)
                        current_slot_array=[]
                        current_slot_tag=word
                else:
                    current_slot_tag = word
                    current_slot_array = []
                    current_slot_array.append(index)
            if (word == 'o' or index == len(pred_list)-1) and current_slot_tag != '':
                slot_arrays[current_slot_tag].append(current_slot_array)
                current_slot_tag=''
                current_slot_array=[]
            if word == 'o':
                continue;
        return slot_arrays;

    def get_slot_metrics(self, preds, golden):
        assert len(preds) == len(golden)


        #initializat dictionary
        slot_tp_tn_fn_counts = {}
        for label in self.slots_label_set.get_labels():
            slot_tp_tn_fn_counts[label+"_fn"]=0
            slot_tp_tn_fn_counts[label+"_fp"]=0
            slot_tp_tn_fn_counts[label+"_tp"]=0       

        for pred, golden_per_query in zip(preds.tolist(), golden.tolist()):
            preds_slot_array = self.create_slot_arrays(pred)
            golden_slot_array = self.create_slot_arrays(golden_per_query)
            query_fn = 0
            query_fp = 0
            for label in self.slots_label_set.get_labels():
                golden_set=set(map(tuple, golden_slot_array[label]))
                preds_set=set(map(tuple, preds_slot_array[label]))

                # use set operator to check each span
                tp_count = len(preds_set & golden_set)
                fp_count = len(preds_set - golden_set)
                fn_count = len(golden_set-preds_set)
                slot_tp_tn_fn_counts[label+"_fn"]=slot_tp_tn_fn_counts[label+"_fn"]+fn_count
                slot_tp_tn_fn_counts[label+"_fp"]=slot_tp_tn_fn_counts[label+"_fp"]+fp_count
                slot_tp_tn_fn_counts[label+"_tp"]=slot_tp_tn_fn_counts[label+"_tp"]+tp_count
                query_fn = query_fn+fn_count
                query_fp = query_fp+fp_count

    
        total_tp_slot = 0;
        total_fp_slot = 0;
        total_fn_slot = 0;
        for label in self.slots_label_set.get_labels():
            total_tp_slot += slot_tp_tn_fn_counts[label+"_tp"];
            total_fp_slot += slot_tp_tn_fn_counts[label+"_fp"];
            total_fn_slot += slot_tp_tn_fn_counts[label+"_fn"];
                
            slot_metric = label+'\t'+ ': total_tp: '+str(slot_tp_tn_fn_counts[label+"_tp"])+', total_fp: '+str(slot_tp_tn_fn_counts[label+"_fp"])+', total_fn: '+str(slot_tp_tn_fn_counts[label+"_fn"]);
            # for debug
            print("slot metric\t{}".format(slot_metric))
            
        overall_precision_slot = 0;
        overall_recall_slot = 0;

        if total_tp_slot != 0:
            overall_precision_slot = total_tp_slot / (total_tp_slot + total_fp_slot);
            overall_recall_slot = total_tp_slot / (total_tp_slot + total_fn_slot);

        return {
             "total_slot_precision": overall_precision_slot,
             "total_slot_recall": overall_recall_slot
             #"slot_f1": f1_score(preds.tolist(), golden.tolist())
        }



        #preds_slot_array = self.create_slot_arrays(self, preds.tolist())
        #golgden_slot_array = self.create_slot_arrays(self, golden.tolist())


        #for pred, golden_per_query in zip(preds.tolist(), golden.tolist()):
        #    golden_set=set(map(tuple, golden_slot_arrays[label]))
        #            prediction_set=set(map(tuple, predicted_slot_arrays[label]))
        #            tp_count = len(prediction_set & golden_set)
        #            fp_count = len(prediction_set-golden_set)
        #            fn_count = len(golden_set-prediction_set)
        #            slot_tp_tn_fn_counts[label+"_fn"]=slot_tp_tn_fn_counts[label+"_fn"]+fn_count
        #            slot_tp_tn_fn_counts[label+"_fp"]=slot_tp_tn_fn_counts[label+"_fp"]+fp_count
        #            slot_tp_tn_fn_counts[label+"_tp"]=slot_tp_tn_fn_counts[label+"_tp"]+tp_count
        #            query_fn = query_fn+fn_count
        #            query_fp = query_fp+fp_count

    # leave for iob
    #def get_slot_metrics(self, preds, golden):
    #    assert len(preds) == len(golden)
    #    return {
    #        "slot_precision": precision_score(preds.tolist(), golden.tolist()),
    #        "slot_recall": recall_score(preds.tolist(), golden.tolist()),
    #        "slot_f1": f1_score(preds.tolist(), golden.tolist())
    #    }

    def compute_metrics(self):
        # checking the length is the same
        assert len(self.intent_preds) == len(self.intent_golden) == len(self.slot_preds) == len(self.slot_golden)
        results = {}

        # library cannot calculate intent, find later
        intent_result = self.get_intent_metrics(self.intent_preds, self.intent_golden)
        slot_result = self.get_slot_metrics(self.slot_preds, self.slot_golden)
        #sementic_result = get_sentence_frame_acc(self.intent_preds, self.intent_golden, self.slot_preds, self.slot_golden)

        results.update(intent_result)
        results.update(slot_result)
        #results.update(sementic_result)

        return results




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
    query = row['MessageText']

    
    intent = row['JudgedIntent']

    slot = row['JudgedConstraints']
	# remove head and end spaces 
    slot = slot.strip()


    # ignore multi turn queries
    conversationContext = row['ConversationContext']

    if  (conversationContext.lower().find('previous') != -1 or 
        conversationContext.lower().find('task') != -1 or 
        conversationContext.lower().find('user') != -1):
        print("multiturn query\t{}\t{}".format(row['ConversationId'], query))
        continue

    # filter invalid intent query
    try:
        intent_label_set.get_ids_from_label(intent.lower())
    except KeyError:
         print("wroing intent query \t{}\t{}".format(query, intent))
         continue

    # invalid query will return empty string
    # here using annotation to extract the real query
    text, tag_string  = slots_label_set.preprocessRawAnnotation(query, slot, useIob=True)

    if text == '' and tag_string == '':
        print("query_with_slot_issue\t{}\t{}".format(query, slot))
        continue

    tokensForQueries.append(text)
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
output_file = '..\\azureml_data\\atis_train.tsv_after_my_preprocessing.tsv'
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



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
# install package for calculation
# https://github.com/chakki-works/seqeval
# only support IOB format....
from seqeval.metrics import precision_score, recall_score, f1_score

'''
import traceback
import warnings
import sys

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback
'''



import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers.modeling_outputs import TokenClassifierOutput
#from transformers import DistilBertPreTrainedModel, DistilBertModel
from transformers import DistilBertTokenizer,DistilBertTokenizerFast
#from transformers import BertForTokenClassification
from transformers import BertPreTrainedModel,BertModel

#from transformers import DistilBertForTokenClassification, AdamW, DistilBertConfig

from transformers import get_linear_schedule_with_warmup
from transformers import BatchEncoding
from tokenizers import Encoding



#########################
# torch tensor operation test below
#########################

output = torch.tensor([[[-0.1221, -0.3479, -0.0684,  0.0110,  0.2062,  0.1621, -0.0185,
           0.0124,  0.5029,  0.7174,  0.2862, -0.2431,  0.1328, -0.1321,
          -0.4819,  0.2302,  0.0615, -0.2918, -0.3064, -0.2910,  0.3069],
         [ 0.0583, -0.2261, -0.1034,  0.1108,  0.0693,  0.2041, -0.1494,
          -0.0275,  0.0287,  0.0484,  0.3371, -0.0232,  0.2029,  0.0085,
          -0.2478,  0.1623,  0.0651, -0.0443, -0.4363, -0.0838,  0.0469],
         [ 0.0049, -0.2861, -0.0598,  0.1419, -0.0236,  0.2379, -0.1346,
          -0.1205,  0.0782,  0.0839,  0.2446,  0.0358,  0.2748, -0.0316,
          -0.1360,  0.1410,  0.0341, -0.1465, -0.4751, -0.1850,  0.0027],
         [-0.2048, -0.2630,  0.1079, -0.1364, -0.0478,  0.2168, -0.2872,
          -0.0433,  0.2038, -0.0138,  0.2251, -0.1251, -0.0528,  0.0448,
          -0.3945,  0.2250,  0.0129,  0.0118, -0.3394,  0.0837,  0.0479],
         [ 0.3469,  0.1035,  0.2902,  0.1426,  0.0818, -0.1072,  0.3632,
           0.0375,  0.3482, -0.2737,  0.0239, -0.1121, -0.1262, -0.2249,
          -0.2885, -0.0055, -0.3414,  0.1324, -0.1100, -0.1086,  0.0097]]])

#version 1
#for i, ele2d in enumerate(output):
#    for j, ele1d in enumerate(ele2d):
#        value, index = torch.max(ele1d, dim=0)
#        print("value: {} anmd index {}".format(value, index))


#version 2
#index = torch.argmax(output.view(-1,21), dim=1)

#print(index)


###############
#local below
###############


import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers.modeling_outputs import TokenClassifierOutput
#from transformers import DistilBertPreTrainedModel, DistilBertModel
from transformers import DistilBertTokenizer,DistilBertTokenizerFast
#from transformers import BertForTokenClassification
from transformers import BertPreTrainedModel,BertModel
#from transformers import DistilBertForTokenClassification, AdamW, DistilBertConfig
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import BatchEncoding
from tokenizers import Encoding

#import horovod.torch as hvd

#from azureml.core import Workspace, Run, Dataset

df = pd.read_csv('E:/azure_ml_notebook/azureml_data/MDM_TrainSet_01202021v1.tsv', sep='\t', encoding="utf-8",
#df = pd.read_csv('E:/azure_ml_notebook/azureml_data/MDM_TrainSet_small_01202021v1.tsv', sep='\t', encoding="utf-8",
#df = pd.read_csv('E:/azure_ml_notebook/azureml_data/MDM_TrainSet_problematic_data.tsv', sep='\t', encoding="utf-8",
#df = pd.read_csv('E:/azure_ml_notebook/azureml_data/atis_test.tsv', sep='\t', encoding="utf-8",
#df = pd.read_csv('E:/azure_ml_notebook/azureml_data/atis_train.tsv', sep='\t', encoding="utf-8",
#df = pd.read_csv('E:/azure_ml_notebook/azureml_data/atis_train_ten.tsv', sep='\t', encoding="utf-8",
    keep_default_na=False,
    dtype={
    'MessageId': object, 'Frequency': object, 'ConversationContext': object, 'SelectionIgnore': object})


#df = pd.read_csv('E:/azure_ml_notebook/azureml_data/atis_train_ten.tsv', sep='\t', encoding="utf-8",
#    keep_default_na=False,
#    dtype={
#    'MessageId': object, 'Frequency': object, 'ConversationContext': object, 'SelectionIgnore': object})


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


#########################
# torch tensor operation test above
#########################


#enc = BertTokenizer.from_pretrained("bert-base-uncased")
fast_tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased') # Load a pre-trained tokenizer

# Tokenizing input text

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# for debug 
print("device: {}".format('cuda' if torch.cuda.is_available() else 'cpu'))

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



# Initializing the model with the torchscript flag
# Flag set to True even though it is not necessary as this model does not have an LM Head.
#config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
#    num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, torchscript=True)

# Instantiating the model
#model = BertModel(config)

# The model needs to be in evaluation mode
#model.eval()



# read label
from typing_extensions import TypedDict
from typing import List,Any
IntList = List[int] # A list of token_ids
IntListList = List[IntList] # A List of List of token_ids, e.g. a Batch


import itertools
class LabelSet:
    def __init__(self, labels: List[str], tokenizer, untagged_id, pad_token_label_id,  useIob=False):
        self.labels_to_id = {}
        self.ids_to_label = {}
        self.untagged_id = untagged_id
        self.labels_to_id["o"] = untagged_id
        self.ids_to_label[untagged_id] = "o"


        self.pad_token_label_id = pad_token_label_id
        self.labels_to_id["pad"] = pad_token_label_id
        self.ids_to_label[pad_token_label_id] = "pad"

        num = 0
        for label in labels:
            # using lower case to compare
            if label.lower() == "o" or label.lower() == "pad":
                print("skip:{}".format(label))
                num = num +1 
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
        return self.ids_to_label[self.untagged_id]

    def get_pad_id(self):
        return self.pad_token_label_id

    def get_pad_label(self):
        return self.ids_to_label[self.pad_token_label_id]

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
    #minic atis
    "PAD",
    "UNK",
    "o",
    "absolute_location",
    "added_text_temp",
    "attachment",
    "attribute_type",
    "audio_device_type",
    "availability",
    "contact_attribute",
    "contact_name",
    "contact_name_type",
    "data_source",
    "data_source_name",
    "data_source_type",
    "date",
    "deck_location",
    "deck_name",
    "destination_calendar",
    "destination_platform",
    "duration",
    "end_date",
    "end_time",
    "feedback_subject",
    "feedback_type",
    "file_action",
    "file_action_context",
    "file_filerecency",
    "file_folder",
    "file_keyword",
    "file_name",
    "file_recency",
    "file_type",
    "from_contact_name",
    "from_relationship_name",
    "implicit_location",
    "job_title",
    "key",
    "meeting_room",
    "meeting_starttime",
    "meeting_title",
    "meeting_type",
    "mergemsg",
    "message",
    "message_category",
    "message_type",
    "move_earlier_time",
    "move_later_time",
    "numerical_increment",
    "office_location",
    "order_ref",
    "org_name",
    "original_contact_name",
    "original_end_date",
    "original_end_time",
    "original_start_date",
    "original_start_time",
    "original_title",
    "people_attribute",
    "phone_number",
    "position_ref",
    "project_name",
    "pronoun",
    "quantity",
    "relationship_name",
    "scenario",
    "search_query",
    "setting_level",
    "setting_type",
    "share_target",
    "sharetarget_name",
    "sharetarget_type",
    "skill_name",
    "slide_content_type",
    "slide_name",
    "slide_number",
    "slot_attribute",
    "source_platform",
    "speed_dial",
    "start_date",
    "start_time",
    "teammeeting_quantifier",
    "teammeeting_starttime",
    "teammeeting_title",
    "teamspace_channel",
    "teamspace_keyword",
    "teamspace_menu",
    "teamspace_tab",
    "teamspace_team",
    "teamsuser_activitytype",
    "teamsuser_status",
    "teamsuser_topic",
    "time",
    "title",
    "to_contact_name",
    "volume_level"
]


# map all slots to lower case
slots_label_set = LabelSet(labels=map(str.lower,slots), 
                            tokenizer =fast_tokenizer, 
                            untagged_id = 2,
                            pad_token_label_id = 0)


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
    "X",
    "accept_meeting",
    "add_attendee",
    "add_contact",
    "add_more",
    "add_to_call",
    "answer_phone",
    "appreciation",
    "assign_nickname",
    "block_time",
    "calendar_notsure",
    "calendar_other",
    "call_back",
    "call_voice_mail",
    "cancel",
    "change_calendar_entry",
    "check_availability",
    "check_im_status",
    "close_setting",
    "communication_other",
    "confirm",
    "connect_to_meeting",
    "contact_meeting_attendees",
    "create_calendar_entry",
    "decline_meeting",
    "delete_calendar_entry",
    "depreciation",
    "devicecontrol_other",
    "disconnect_from_meeting",
    "end_call",
    "feedback_other",
    "file_download",
    "file_navigate",
    "file_open",
    "file_other",
    "file_search",
    "file_share",
    "find_calendar_entry",
    "find_calendar_entry_followup",
    "find_calendar_when",
    "find_calendar_where",
    "find_calendar_who",
    "find_calendar_why",
    "find_contact",
    "find_duration",
    "find_meeting_insight",
    "find_meeting_room",
    "finish_task",
    "forwarding_off",
    "forwarding_on",
    "forwarding_status",
    "get_notifications",
    "go_back",
    "go_forward",
    "goto_slide",
    "help",
    "hide_whiteboard",
    "hold",
    "ignore_incoming",
    "ignore_with_message",
    "lock",
    "make_call",
    "mark",
    "mark_tentative",
    "mute",
    "mute_participant",
    "navigate_calendar",
    "next_slide",
    "open_setting",
    "other",
    "press_key",
    "previous_slide",
    "query_last_text",
    "query_message",
    "query_sender",
    "query_speeddial",
    "redial",
    "reject",
    "repeat",
    "repeat_slowly",
    "repeat_user",
    "reply",
    "resume",
    "retry",
    "search_messages",
    "search_org_chart",
    "search_people_attribute",
    "search_people_by_attribute",
    "search_people_by_name",
    "select_any",
    "select_item",
    "select_more",
    "select_none",
    "select_other",
    "send_text",
    "send_text_meeting",
    "set_default_device",
    "set_speeddial",
    "set_volume",
    "show_next",
    "show_previous",
    "show_whiteboard",
    "slide_back",
    "speakerphone_off",
    "speakerphone_on",
    "start_over",
    "start_presenting",
    "stop",
    "stop_presenting",
    "submit_feedback",
    "teamsaction_other",
    "teamspace_addtoteam",
    "teamspace_checkmember",
    "teamspace_createteam",
    "teamspace_favorite",
    "teamspace_follow",
    "teamspace_help",
    "teamspace_jointeam",
    "teamspace_navigate",
    "teamspace_removemember",
    "teamspace_search",
    "teamspace_sharechannel",
    "teamspace_sharetab",
    "teamspace_showhotkey",
    "teamspace_showtab",
    "teamspace_unfavorite",
    "teamspace_unfollow",
    "teamsuser_checkorg",
    "teamsuser_openchat",
    "teamsuser_setstatus",
    "teamsuser_showactivity",
    "time_remaining",
    "transfer",
    "turn_down",
    "turn_up",
    "unmute",
    "volume_down",
    "volume_up"
]

intent_label_set = IntentLabelSet(labels=map(str.lower,intents))

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


    # for iob
    def get_intent_metrics_Iob(self, preds, golden):
        #acc = (preds == golden).mean()
        #return {
        #    "intent_acc": acc
        #}
        assert len(preds) == len(golden)

        acc = (preds == golden).mean()

        # repo :
        #  origginal preds dtype=int64, nparray
        # inside is labelid
        #[0:4478]
        # originla golden (output_intent_labes inds), dtype=int64, nparray
        # [0:4478]
        # inside is labelid

       
        return {
            # originla code using this
            "intent_acc": acc
            # ? belo code cannot work, need to sutdy
            # 
            #"intent_precision": precision_score(preds, golden),
            #"intent_recall": recall_score(preds, golden)
        }


    def get_intent_metrics(self, preds, golden):


        if self.useIob is True:
            return self.get_intent_metrics_Iob(preds, golden)

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


    # leave for iob
    '''
    def get_slot_metrics_Iob(self, preds, golden):
        assert len(preds) == len(golden)



        # repo :
        #  origginal slot_preds_list 2d list,
        # inside is slot, not id, so need to transformed
        # for each element, length is not fixed length
        # ? but in my case will the same , should be ok


        # map :
        # https://stackoverflow.com/questions/42594695/how-to-apply-a-function-map-values-of-each-element-in-a-2d-numpy-array-matrix
        def myfunc(z):
            return self.slots_label_set.get_label(z)

        #temp = myfunc(preds)
        preds_labels = np.vectorize(myfunc)(preds)

        # 
        #[0:4478]
        # originla golden (output_slot_labellist),2d list,
        # inside is slot, not id
        # for each element, length is not fixed length
        # ? but in my case will the same , should be ok

        golden_labels = np.vectorize(myfunc)(golden)



        return {
            "slot_precision": precision_score(golden_labels.tolist(), preds_labels.tolist()),
            "slot_recall": recall_score(golden_labels.tolist(), preds_labels.tolist()),
            "slot_f1": f1_score(golden_labels.tolist(), preds_labels.tolist())
        }
    '''


    def get_slot_metrics_Iob(self, preds, golden):
        assert len(preds) == len(golden)

        # map list of id to label
        preds_labels_list = [[] for _ in range(len(preds))]
        golden_labels_list = [[] for _ in range(len(golden))]

    
        for i in range(len(golden)):
            for j in range(len(golden[i])):
                preds_labels_list[i].append(slots_label_set.get_label(preds[i][j]))
                golden_labels_list[i].append(slots_label_set.get_label(golden[i][j]))

        return {
            "slot_precision": precision_score(golden_labels_list, preds_labels_list),
            "slot_recall": recall_score(golden_labels_list, preds_labels_list),
            "slot_f1": f1_score(golden_labels_list,  preds_labels_list)
        }

    def get_slot_metrics(self, preds, golden):

        if self.useIob is True:
            return self.get_slot_metrics_Iob(preds, golden)


        assert len(preds) == len(golden)


        #initializat dictionary
        slot_tp_tn_fn_counts = {}
        for label in self.slots_label_set.get_labels():
            slot_tp_tn_fn_counts[label+"_fn"]=0
            slot_tp_tn_fn_counts[label+"_fp"]=0
            slot_tp_tn_fn_counts[label+"_tp"]=0       

        #for pred, golden_per_query in zip(preds.tolist(), golden.tolist()):
        for pred, golden_per_query in zip(preds, golden):
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


    # leave it no need since overloading 
    #def compute_metrics_IOB(self):
    #    # checking the length is the same
    #    assert len(self.intent_preds) == len(self.intent_golden) == len(self.slot_preds) == len(self.slot_golden)
    #    results = {}

    #    intent_result = self.get_intent_metrics(self.intent_preds, self.intent_golden)
    #    slot_result = self.get_slot_metrics(self.slot_preds, self.slot_golden)
    #    #sementic_result = get_sentence_frame_acc(intent_preds, intent_labels, slot_preds, slot_labels)

    #    results.update(intent_result)
    #    results.update(slot_result)
    #    #results.update(sementic_result)

    #    return results

    def compute_metrics(self, ignore_pad=False):


        #if self.useIob is True:
        #    return self.compute_metrics_IOB()

        # checking the length is the same
        assert len(self.intent_preds) == len(self.intent_golden) == len(self.slot_preds) == len(self.slot_golden)
        results = {}

        slot_preds_list = [[] for _ in range(self.slot_golden.shape[0])]
        slot_golden_list = [[] for _ in range(self.slot_golden.shape[0])]

        for i in range(self.slot_golden.shape[0]):
            for j in range(self.slot_golden.shape[1]):
                # v2
                 # ignore pad_token_label_id
                if ignore_pad == True:
                    if self.slot_golden[i, j] != self.slots_label_set.get_pad_id():
                        slot_preds_list[i].append(self.slot_preds[i][j])
                        slot_golden_list[i].append(self.slot_golden[i][j])
                else:
                        slot_preds_list[i].append(self.slot_preds[i][j])
                        slot_golden_list[i].append(self.slot_golden[i][j])


                #  no need to map label
                # it will be done inside get_slot_metrics()
                # there is np.array mapping inside get_slot_metrics()
                #if self.slot_golden[i, j] != self.slots_label_set.get_pad_label():
                    #slot_preds_wo_pad[i].append(slots_label_set.get_label(self.slot_preds[i][j]))
                    #slot_golden_wo_pad[i].append(slots_label_set.get_label(self.slot_golden[i][j]))




        intent_result = self.get_intent_metrics(self.intent_preds, self.intent_golden)
        slot_result = self.get_slot_metrics(slot_preds_list, slot_golden_list)
        #sementic_result = get_sentence_frame_acc(self.intent_preds, self.intent_golden, self.slot_preds, self.slot_golden)

        results.update(intent_result)
        results.update(slot_result)
        #results.update(sementic_result)

        return results


        #if ignore_pad == True:

        #else:



        #    # library cannot calculate intent, find later
        #    intent_result = self.get_intent_metrics(self.intent_preds, self.intent_golden)
        #    slot_result = self.get_slot_metrics(self.slot_preds, self.slot_golden)
            #sementic_result = get_sentence_frame_acc(self.intent_preds, self.intent_golden, self.slot_preds, self.slot_golden)

        #    results.update(intent_result)
        #    results.update(slot_result)
        #    #results.update(sementic_result)

        #    return results


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
intent_labels = []
# iterative get labele and also append padding based on text_ids
labels_for_text_ids = []
for i, row in df.iterrows():
    

    #for debug
    #print("conversation id {}".format(row['ConversationId']))

    query = row['MessageText']

    
    intent = row['JudgedIntent']

    slot = row['JudgedConstraints']
	# remove head and end spaces 
    slot = slot.strip()



    # ignore any empty queries
    if len(query.strip()) == 0 or len(intent.strip()) == 0 or len(slot) == 0:
        print("empty query skipped\t{}".format(row['ConversationId']))
        continue


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
    text, tag_string  = slots_label_set.preprocessRawAnnotation(query, slot)

    if text == '' and tag_string == '':
        print("query_with_slot_issue\t{}\t{}".format(query, slot))
        continue

    #append labels for [CLS] / [SEP] to tag_string
    #tag_string =  slots_label_set.get_untagged_label() + ' '+ tag_string + ' ' + slots_label_set.get_untagged_label()
    # v1: 
    # CLS , SEP label = 0
    # B-label extend 
    tag_string =  slots_label_set.get_pad_label() + ' '+ tag_string + ' ' + slots_label_set.get_pad_label()


    aligned_label_ids = slots_label_set.get_aligned_label_ids_from_aligned_label(
        map(str.lower,tag_string.split())
    )

    if None in set(aligned_label_ids):
        print("query_with_unkonwn slot_issue\t{}\t{}".format(query, slot))
        continue

    # only if it is valid string for slot then add intent label
    # using low case slot to lookup
    intent_labels.append(intent_label_set.get_ids_from_label(intent.lower()))


    # replcae by class's output word string
    text_id = fast_tokenizer.encode(text, max_length=300, padding='max_length', truncation=True)
    text_ids.append(text_id)





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
            #labels_for_text_id.append(slots_label_set.get_untagged_id())
            # padding add pad id
            labels_for_text_id.append(slots_label_set.get_pad_id())
    labels_for_text_ids.append(labels_for_text_id)


    # for debug
    #print("text:{}".format(text))
    #print("text id :{}".format(text_id))
    #print("slot: {}".format(labels_for_text_id))



num_slot_labels = len(set(slots_label_set.get_labels()))
num_intent_labels = len(set(intent_label_set.get_labels()))

### for debug
#print('text_ids[0]: {}'.format(text_ids[0]))
#print('labels_for_text_ids[0]: {}'.format(labels_for_text_ids[0]))
#print('att mask[0] : {}'.format(att_masks[0]))
#print('text_ids[3]: {}'.format(text_ids[3]))
#print('labels_for_text_ids[3]: {}'.format(labels_for_text_ids[3]))
#print('att mask[3] : {}'.format(att_masks[3]))
#print('text_ids[4]: {}'.format(text_ids[4]))
#print('labels_for_text_ids[4]: {}'.format(labels_for_text_ids[4]))
#print('att mask[4] : {}'.format(att_masks[4]))


#sklearn split data
#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
#? same random state but in different ways, how to make sure each query is aligned
#  https://www.cnblogs.com/Yanjy-OnlyOne/p/11288098.html
# it seems same random_state will generate the same result
#train_x, test_val_x, train_y, test_val_y = train_test_split(text_ids, labels_for_text_ids, random_state=111, test_size=0.2)
#train_m, test_val_m = train_test_split(att_masks, random_state=111, test_size=0.2)
#test_x, val_x, test_y, val_y = train_test_split(test_val_x, test_val_y, random_state=111, test_size=0.5)
#test_m, val_m = train_test_split(test_val_m, random_state=111, test_size=0.5)


# make traning data / test data / validation data the same
# including intent and slot
# intent first then slot
train_x, train_y, train_z = text_ids, intent_labels, labels_for_text_ids,
train_m = att_masks
test_x, test_y, test_z = text_ids, intent_labels, labels_for_text_ids
test_m = att_masks
val_x, val_y, val_z = text_ids, intent_labels, labels_for_text_ids
val_m = att_masks



# Convert all inputs and labels into torch tensors, the required datatype 
#https://pytorch.org/docs/stable/tensors.html
# can be multiple dimentionas
train_x = torch.tensor(train_x)
test_x = torch.tensor(test_x)
val_x = torch.tensor(val_x)
train_y = torch.tensor(train_y)
test_y = torch.tensor(test_y)
val_y = torch.tensor(val_y)
train_z = torch.tensor(train_z)
test_z = torch.tensor(test_z)
val_z = torch.tensor(val_z)
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

print('train_z dimen {}'.format(train_z.shape))
print('test_z dimen {}'.format(test_z.shape))
print('train_z dimen {}'.format(train_z.shape))
print('val_z dimen {}'.format(val_z.shape))

print('train_m dimen {}'.format(train_m.shape))
print('test_m dimen {}'.format(test_m.shape))
print('train_m dimen {}'.format(train_m.shape))
print('val_m dimen {}'.format(val_m.shape))



###############
# check gpu below
###############

gpu_available = torch.cuda.is_available()


###############
#  check gpu above
###############


###############
#local training data setup in learning below
###############

# kwargs = {'num_workers': 1, 'pin_memory': True} if gpu_available else {}
#batch_size = 32
batch_size = 6
# https://pytorch.org/docs/stable/data.html
# for local remove repliaces rank optional arigment
# also remove distributedSampler
#train_data = TensorDataset(train_x, train_m, train_y)
#train_dataloader = DataLoader(train_data, sampler=None, batch_size=batch_size)
train_data = TensorDataset(train_x, train_m, train_y, train_z)
train_dataloader = DataLoader(train_data, sampler=None, batch_size=batch_size)
#val_data = TensorDataset(val_x, val_m, val_y)
#val_dataloader = DataLoader(val_data, sampler=None, batch_size=batch_size)
val_data = TensorDataset(val_x, val_m, val_y, val_z)
val_dataloader = DataLoader(val_data, sampler=None, batch_size=batch_size)

###############
#local training data setup in learning above
###############



#model = DistilBertForTokenClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels,
#                                                            output_attentions=False, output_hidden_states=False)

'''
#class DistilBertForTokenClassificationFilesDomain(DistilBertPreTrainedModel):
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

        # originla value if return_dict is not none
        #output = (logits,) + outputs[1:]
        #return ((loss,) + output) if loss is not None else output

        # originla value if return_dict is none
        #return TokenClassifierOutput(
        #    loss=loss,
        #    logits=logits,
        #    hidden_states=outputs.hidden_states,
        #    attentions=outputs.attentions,
        #)



        #version 1 - runtime error
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

        # version 3
        #slot_label_tensor = torch.argmax(logits.view(-1, self.num_labels), dim=1)
        #return ((loss,) + output) 
'''       

'''
class DistilBertForTokenClassificationFilesDomain(BertForTokenClassification):

    def forward(self, input_ids, attention_mask=None, labels=None):
        sequence_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output);


        # replace my label function
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


        slot_output = torch.argmax(logits, -1);
        return loss, slot_output;
'''

'''
class DistilBertForTokenClassificationFilesDomain(BertForTokenClassification):

    def forward(self, input_ids, attention_mask=None, labels=None):
        sequence_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output);
        # for training 
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss

            
            #yue's calculation

            ##attention_mask_label = None
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1;
                active_logits = logits.view(-1, self.num_labels)[active_loss];
                active_labels = labels.view(-1)[active_loss];
                loss = loss_fct(active_logits, active_labels);
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1));

            #hugging face calculation
            # Only keep active parts of the loss
            #if attention_mask is not None:
            #    active_loss = attention_mask.view(-1) == 1
            #    active_logits = logits.view(-1, self.num_labels)
            #    active_labels = torch.where(
            #        active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
            #    )
            #    loss = loss_fct(active_logits, active_labels)
            #else:
            #    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            return loss;

        else:
        # for inference
            slot_output = torch.argmax(logits, -1);
            return slot_output;
'''

class IntentClassifier(nn.Module):
    def __init__(self, input_dim, num_intent_labels, dropout_rate=0.):
        super(IntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_intent_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class SlotClassifier(nn.Module):
    def __init__(self, input_dim, num_slot_labels, dropout_rate=0.):
        super(SlotClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_slot_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class DistilBertForTokenClassificationFilesDomain(BertPreTrainedModel):


    def __init__(self, config, num_intent_labels, num_slot_labels, weight=None, ):
        super(DistilBertForTokenClassificationFilesDomain, self).__init__(config)
        # pretrained model (not specifific to slot intent do not need labels)
        # self.num_labels = config.num_labels
        self.num_intent_labels = num_intent_labels
        self.num_slot_labels = num_slot_labels

        # ? might be no need this so comment it 
        # yue does not have this
        #self.weight = weight

        # yue is using BertModel for class but internet is using BertPreTrainedModel
        # ? need to check what is difference
        self.bert = BertModel(config=config)  # Load pretrained bert


        # one class for intent
        # one class for slot
        # using bert's dropout_rate and hideen dimention
        # ? need to make sure config has dropout_rate
        self.intent_classifier = IntentClassifier(config.hidden_size, self.num_intent_labels, config.hidden_dropout_prob)
        self.slot_classifier = SlotClassifier(config.hidden_size, self.num_slot_labels, config.hidden_dropout_prob)

        # ? need to study what this function is for
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, intent_label_ids=None, slot_label_ids=None):


        # remove token_type_ids since it is not neceessary
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = output[0]
        pooled_output = output[1]  # [CLS]

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)

        # define intent loss / slot loss 
        total_intent_loss = 0
        # 1. Intent Softmax
        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1))
            else:
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1))
            total_intent_loss += intent_loss

        # 2. Slot Softmax
        # ignore coefficeint part
        total_slot_loss = 0
        if slot_label_ids is not None:
            # remove ignore_index since unnecessary
            #slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
            slot_loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                active_labels = slot_label_ids.view(-1)[active_loss]
                slot_loss = slot_loss_fct(active_logits, active_labels)
            else:
                slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_label_ids.view(-1))
            # ignore coefficeint part
            #total_slot_loss += self.args.slot_loss_coef * slot_loss
            total_slot_loss += slot_loss


        # if either one is none, then do inference
        if intent_label_ids is None or slot_label_ids is None:
            # output intent label as weight
            intent_output = torch.argmax(intent_logits, -1);
            slot_output = torch.argmax(slot_logits, -1);

            # QAS
            # for domain, it needs to have probabilty
            #    files_enus_mv7_domain_svm_score (tag: 1, string: 0)
            #            0[-1,-1]=0.1968164

            # for intent
            #                files_enus_mv7_intent_svm_score (tag: 11, string: 0)
            #            0[-1,-1]=2.2269628
            #            1[-1,-1]=-1.5521804
            #            2[-1,-1]=-1.6113045
            #            3[-1,-1]=-1.0233102
            #            4[-1,-1]=-1.3408698
            #            5[-1,-1]=-1.8470569
            #            6[-1,-1]=-1.0570247
            #            7[-1,-1]=-1.167596
            #            8[-1,-1]=-1.8137677
            #            9[-1,-1]=-0.99997
            #            10[-1,-1]=-1.389961


            intent_prob = intent_logits.softmax(dim=1)
            #indices = torch.tensor([1]).to(device)
            # this is for domain the second since in domian = 1
            #temp = torch.index_select(intent_logits, 1, indices),

            #return intent_output, slot_output
            return intent_output, slot_output, intent_prob
        else:
            return total_intent_loss, total_slot_loss



'''
#Here we instantiate our model class. 
#We use a compact version, that is trained through model distillation from a base BERT model and modified to include a classification layer at the output. This compact version has 6 transformer layers instead of 12 as in the original BERT model.
# this class class DistilBertForSequenceClassification(DistilBertPreTrainedModel):
# 
#https://huggingface.co/transformers/v1.2.0/_modules/pytorch_transformers/modeling_distilbert.html
# replace with my defined class but the same parameters
model = DistilBertForTokenClassificationFilesDomain.from_pretrained('distilbert-base-uncased', num_labels=num_labels,
                                                            output_attentions=False, output_hidden_states=False)

'''



##################################################
# load TNLR model below
##################################################

'''
output_dir = '../TNLR/'
import os, argparse
# if folder does not exist then create
if not os.path.exists(output_dir):
    os.makedirs(output_dir)




# save your training arguments together with the trained model
# originlal those are from input argument
# here setup value to mimic
# learning_rate = args.learning_rate
# adam_epsilon = args.adam_epsilon

# load model back
# you need to know exact class for each one
# it expects file like pytorch_model.bin
# <failed >version 1: so rename *pt to see if it wokrs. it will complaint 
#torch.nn.modules.module.ModuleAttributeError: 'DistilBertForTokenClassification' object has no attribute 'keys'
# version2 : load pytroch_model.bin (not sure if it is after trained or not)
#model = DistilBertForTokenClassification.from_pretrained(output_dir)
#model = DistilBertForTokenClassificationFilesDomain.from_pretrained(output_dir)


from transformers import BertConfig;
bert_config = BertConfig();
# no need to provide level for bertPreTrainModel
#bert_config.num_labels = num_labels;
bert_config.num_hidden_layers = 3
bert_config.output_attentions = False;
bert_config.output_hidden_states = False;



#model = DistilBertForTokenClassificationFilesDomain.from_pretrained(output_dir+'tnlrv3-base.pt', num_labels=num_labels,
#                                                            output_attentions=False, output_hidden_states=False)

# minic yue is ok to load
# if no providing config, it will fail...
# it seems config.json it not important and it can still be loaded
#model = DistilBertForTokenClassificationFilesDomain.from_pretrained(output_dir+'tnlrv3-base.pt')
#model = DistilBertForTokenClassificationFilesDomain.from_pretrained(output_dir+'tnlrv3-base.pt', config=bert_config)


model = DistilBertForTokenClassificationFilesDomain.from_pretrained(output_dir+'tnlrv3-base.pt', 
    config=bert_config,
    num_intent_labels=num_intent_labels,
    num_slot_labels=num_slot_labels)

# load tokenizer back
# you need to know exact class for each one
#fast_tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased') # Load a pre-trained tokenizer

# copy model to  GPU/CPU to work
model.to(device)


print("load model done: !")
'''

##################################################
# load TNLR model above
##################################################



##################################################
# load local cpu-based pretrained model below
##################################################


#output_dir = './outputs/'

import os, argparse
# if folder does not exist then create
if not os.path.exists(output_dir):
    os.makedirs(output_dir)




# save your training arguments together with the trained model
# originlal those are from input argument
# here setup value to mimic
# learning_rate = args.learning_rate
# adam_epsilon = args.adam_epsilon

# load model back
# you need to know exact class for each one
# it expects file like pytorch_model.bin
# <failed >version 1: so rename *pt to see if it wokrs. it will complaint 
#torch.nn.modules.module.ModuleAttributeError: 'DistilBertForTokenClassification' object has no attribute 'keys'
# version2 : load pytroch_model.bin (not sure if it is after trained or not)
#model = DistilBertForTokenClassification.from_pretrained(output_dir)
#model = DistilBertForTokenClassificationFilesDomain.from_pretrained(output_dir)


from transformers import BertConfig;
bert_config = BertConfig();
# no need to provide level for bertPreTrainModel
#bert_config.num_labels = num_labels;
bert_config.num_hidden_layers = 3
bert_config.output_attentions = False;
bert_config.output_hidden_states = False;



#model = DistilBertForTokenClassificationFilesDomain.from_pretrained(output_dir+'tnlrv3-base.pt', num_labels=num_labels,
#                                                            output_attentions=False, output_hidden_states=False)

# minic yue is ok to load
# if no providing config, it will fail...
# it seems config.json it not important and it can still be loaded
#model = DistilBertForTokenClassificationFilesDomain.from_pretrained(output_dir+'tnlrv3-base.pt')
#model = DistilBertForTokenClassificationFilesDomain.from_pretrained(output_dir+'tnlrv3-base.pt', config=bert_config)



model = DistilBertForTokenClassificationFilesDomain.from_pretrained(output_dir+'pytorch_model.bin', 
        config=bert_config,
        num_intent_labels=num_intent_labels,
        num_slot_labels=num_slot_labels)

# load tokenizer back
# you need to know exact class for each one
#fast_tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased') # Load a pre-trained tokenizer

# copy model to  GPU/CPU to work
model.to(device)


print("load model done: !")


##################################################
# load local cpu-based pretrained model below
##################################################


##################################################
# load trained onnx model (CPU)
##################################################
'''
#https://github.com/onnx/tutorials
# https://www.onnxruntime.ai/python/auto_examples/plot_load_and_predict.html#


# need to 'pip install onnx'
# https://thenewstack.io/tutorial-using-a-pre-trained-onnx-model-for-inferencing/
import onnx
import onnxruntime
# https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html

model = onnx.load("E:\\azure_ml_notebook\\outputs_temp_load_v1_02142021v1\\traced_distill_bert.onnx.bin")
onnx.checker.check_model(model)
'''


##################################################
# training model below, not yet finisihed, just placeholder
##################################################


print("training model ....")

test_input_tensor = torch.tensor([[101, 12453, 19453, 6254, 102]])
at_mask_tensor = torch.tensor([[1, 1, 1, 1, 1]])

intent_label_mask_tensor = torch.tensor([[1]])
slot_label_mask_tensor = torch.tensor([[0, 0, 6, 0, 0]])
with torch.no_grad():
    model.eval()

    # if going with seperate outputs
    intent_loss,slot_loss = model(test_input_tensor, attention_mask=at_mask_tensor, 
        intent_label_ids=intent_label_mask_tensor,
        slot_label_ids=slot_label_mask_tensor
        )

    print("intent_loss: {}".format(intent_loss))
    print("slot_loss: {}".format(slot_loss))

print("training model done")


##################################################
# training model above
##################################################


##################################################
# evaluate model - single query below
##################################################
'''
print("evaluate model ....")

test_input_tensor = torch.tensor([[101, 12453, 19453, 6254, 102]])
at_mask_tensor = torch.tensor([[1, 1, 1, 1, 1]])
# inference no need label
#intent_label_mask_tensor = torch.tensor([[1]])
#slot_label_mask_tensor = torch.tensor([[0, 0, 6, 0, 0]])
with torch.no_grad():
    model.eval()


    # if going with seperate outputs
    #intent_output,slot_output = model(test_input_tensor, attention_mask=at_mask_tensor
    #    )
    intent_output,slot_output,intent_prob = model(test_input_tensor, attention_mask=at_mask_tensor
        )

    print("intent_output: {}".format(intent_output))
    print("slot_output: {}".format(slot_output))
    #print("intent_output: {}".format(intent_prob))

print("evaluate model done")
'''

##################################################
# evaluate model - single query above
##################################################

##################################################
# evaluate model - a file below - not onnx model
##################################################

# initialize evaluation test object
evaluation_test = Evaluation(slots_label_set, intent_label_set)

with torch.no_grad():	
    # Put the model in evaluation mode--the dropout layers behave differently	
    # during evaluation.	
    model.eval()	
        	
    for k, (mb_x, mb_m, mb_y, mb_z) in enumerate(val_dataloader):	
        mb_x = mb_x.to(device)	
        mb_m = mb_m.to(device)	
        mb_y = mb_y.to(device)	
        mb_z = mb_z.to(device)		
        intent_output,slot_output,intent_prob = model(mb_x, attention_mask=mb_m)	




        evaluation_test.add_intent_pred_and_golden(intent_output, mb_y)	
        evaluation_test.add_slot_pred_and_golden(slot_output, mb_z)	
        # for debug	
        #print('evaluate label result {}'.format(slot_output))	

    # my calculation        	
    print(' Validation metric : {}'.format(evaluation_test.compute_metrics()))
    print(' Validation metric wo pad: {}'.format(evaluation_test.compute_metrics(ignore_pad=True)))


##################################################
# evaluate model - a file above - not onnx model
##################################################

##################################################
# evaluate model - a file below - oxxn -  not yet success
##################################################
'''
# initialize evaluation test object
evaluation_test = Evaluation(slots_label_set, intent_label_set)
evaluation_test_iob = Evaluation(slots_label_set, intent_label_set, useIob=True)

for k, (mb_x, mb_m, mb_y, mb_z) in enumerate(val_dataloader):	
    mb_x = mb_x.to(device)	
    mb_m = mb_m.to(device)	
    mb_y = mb_y.to(device)	
    mb_z = mb_z.to(device)		
    #intent_output,slot_output,intent_prob = model(mb_x, attention_mask=mb_m)	
    session = onnxruntime.InferenceSession(model)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name



    evaluation_test.add_intent_pred_and_golden(intent_output, mb_y)	
    evaluation_test.add_slot_pred_and_golden(slot_output, mb_z)
    evaluation_test_iob.add_intent_pred_and_golden(intent_output, mb_y)	
    evaluation_test_iob.add_slot_pred_and_golden(slot_output, mb_z)		



# my calculation        	
#print(' Validation metric : {}'.format(evaluation_test.compute_metrics()))
# IOB calculation
print(' Validation metric Iob: {}'.format(evaluation_test_iob.compute_metrics()))
'''

##################################################
# evaluate model - a file above - oxxn -  not yet success
##################################################



##################################################
# Store model(not in pytorch) 
##################################################

# ussing distill bert 
'''
#torch.onnx.export(model, dummy_input, 'traced_distill_bert.onnx', verbose=True)
'''

#follow yue's suggestion to add output
# # ouput is slightly different, not sure it is related to 'do_constant_folding' for optimization for other parameter 
'''
torch.onnx.export(model=model,
    args=(dummy_input),
    f='traced_distill_bert.onnx.bin',
    input_names = ["input_ids"],
    verbose=True,
    #output_names = ["intent_output", "slot_output"],
    output_names = ['intent_output', 'slot_output','intent_prob'],
    do_constant_folding = True,
    opset_version=11,
    #dynamic_axes = {'input_ids': {1: '?'}, 'intent_output': {1: '?'}, 'slot_output': {1: '?'}}
    dynamic_axes = {'input_ids': {1: '?'}, 'intent_output': {1: '?'}, 'slot_output': {1: '?'},'intent_prob': {1: '?'}}
    )
'''


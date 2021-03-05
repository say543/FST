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

df = pd.read_csv('E:/azure_ml_notebook/azureml_data/MDM_TrainSet_01202021v1.tsv', sep='\t', encoding="utf-8",
#df = pd.read_csv('E:/azure_ml_notebook/azureml_data/MDM_TrainSet_ten_01202021v1.tsv', sep='\t', encoding="utf-8",
#df = pd.read_csv('E:/azure_ml_notebook/azureml_data/MDM_TrainSet_double_fake_annotation_01202021v1.tsv', sep='\t', encoding="utf-8",
#df = pd.read_csv('E:/azure_ml_notebook/azureml_data/MDM_TrainSet_problematic_data.tsv', sep='\t', encoding="utf-8",
#df = pd.read_csv('E:/azure_ml_notebook/azureml_data/atis_test.tsv', sep='\t', encoding="utf-8",
#df = pd.read_csv('E:/azure_ml_notebook/azureml_data/atis_train.tsv', sep='\t', encoding="utf-8",
#df = pd.read_csv('E:/azure_ml_notebook/azureml_data/atis_train_ten.tsv', sep='\t', encoding="utf-8",
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
    def __init__(self, labels: List[str], outofdomain_id):
        self.labels_to_id = {}
        self.ids_to_label = {}

        self.outofdomain_id = outofdomain_id
        self.labels_to_id["x"] = outofdomain_id
        self.ids_to_label[outofdomain_id] = "x"

        num = 0
        for label in labels:
            if label.lower() == "x":
                print("skip:{}".format(label))
                num = num +1 
                continue
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

    def get_outofdomain_id(self):
        return self.labels_to_id["x"]

    def get_outofdomain_label(self):
        return self.ids_to_label[self.outofdomain_id]

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

intents_map_to_outofdomain_intent = {
    'other_domain',
    'notsure_other',
    'non_sense',
    'ambiguous',
    'multi_intent',
    'communication_notsure'
}

intent_label_set = IntentLabelSet(labels=map(str.lower,intents), outofdomain_id = 0)

class Evaluation():
    def __init__(self, slots_label_set, intent_label_set, useIob=False):

        # here IOB means IOB2 format
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

        # map from id to label
        #for i in range(len(golden)):
        #    for j in range(len(golden[i])):
        #        preds_labels_list[i].append(slots_label_set.get_label(preds[i][j]))
        #        golden_labels_list[i].append(slots_label_set.get_label(golden[i][j]))

        # map from id to label
        for i in range(len(golden)):
            # using previous token to decide whether it should be B or I
            # map label form lower case to upper case for seqeval pacakage
            prev_pred_label = ""
            prev_golden_label = ""
            for j in range(len(golden[i])):

                pred_label = slots_label_set.get_label(preds[i][j])
                golden_label = slots_label_set.get_label(golden[i][j])

                
                # if new label start
                if pred_label == slots_label_set.get_untagged_label() or pred_label == slots_label_set.get_pad_label():
                    preds_labels_list[i].append(pred_label.upper())
                elif pred_label != prev_pred_label:
                    preds_labels_list[i].append('B-'+ pred_label.upper())
                else:
                    preds_labels_list[i].append('I-'+ pred_label.upper())
                prev_pred_label = pred_label

                # if new label start
                if golden_label == slots_label_set.get_untagged_label() or golden_label == slots_label_set.get_pad_label():
                    golden_labels_list[i].append(golden_label.upper())
                elif golden_label != prev_golden_label:
                    golden_labels_list[i].append('B-'+ golden_label.upper())
                else:
                    golden_labels_list[i].append('I-'+ golden_label.upper())
                prev_golden_label = golden_label


        ret_dic = {}
        classification_dic = classification_report(golden_labels_list, preds_labels_list, output_dict=True)

        for key, value in classification_dic.items():

            #micro avg       0.50      0.50      0.50         2
            #macro avg       0.50      0.50      0.50         2
            #weighted avg       0.50      0.50      0.50         2
            # ignore these three keys

            if key != 'micro avg' and key != 'macro avg' and key != 'weighted avg':
                ret_dic[key] = value

        ret_dic['slot_precision'] = precision_score(golden_labels_list, preds_labels_list, average=None)
        ret_dic['slot_recall'] = recall_score(golden_labels_list, preds_labels_list, average=None)
        ret_dic['slot_f1'] = f1_score(golden_labels_list,  preds_labels_list, average=None)


        return ret_dic
        #return {
        #    "classfication_report": classification_report(golden_labels_list, preds_labels_list),
        #    "slot_precision": precision_score(golden_labels_list, preds_labels_list, average=None),
        #    "slot_recall": recall_score(golden_labels_list, preds_labels_list, average=None),
        #    "slot_f1": f1_score(golden_labels_list,  preds_labels_list, average=None)
        #}

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




# iterative get labele and also append padding based on text_ids
tokensForQueries = []
labelsForQueries =[]
labelIdsForQueries = []


text_ids = []
att_masks = []
intent_labels = []
# iterative get labele and also append padding based on text_ids
labels_for_text_ids = []

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

    # ignore any empty queries
    if len(query.strip()) == 0 or len(intent.strip()) == 0 or len(slot) == 0:
        print("empty query skipped\t{}".format(row['ConversationId']))
        continue

    # ignore multi turn queries
    conversationContext = row['ConversationContext']

    if  (conversationContext.lower().find('previousturndomain') != -1 or 
        conversationContext.lower().find('previousturnintent') != -1 or 
        conversationContext.lower().find('taskframeentitystates') != -1 or
        conversationContext.lower().find('taskframeguid') != -1 or
        conversationContext.lower().find('taskframename') != -1 or
        conversationContext.lower().find('taskframestatus') != -1
        # leave presonal grammar as first turn as well even though they are not being used
        #conversationContext.lower().find('usercontacts') != -1 or 
        #conversationContext.lower().find('userfilenames') != -1 or 
        #conversationContext.lower().find('userfilenameskeyphrases') != -1 or 
        #conversationContext.lower().find('Usermeetingsubjects') != -1
        ):
        print("multiturn query\t{}\t{}".format(row['ConversationId'], query))
        continue

    # filter invalid intent query
    try:
        intent_label_set.get_ids_from_label(intent.lower())
    except KeyError:
        # if not following cases, igonre them
        # otherwuse, set then as x
        if intent.lower() not in intents_map_to_outofdomain_intent:
            print("wroing intent query \t{}\t{}".format(query, intent))
            continue
        intent = intent_label_set.get_outofdomain_label()

    # invalid query will return empty string
    # here using annotation to extract the real query
    text, tag_string  = slots_label_set.preprocessRawAnnotation(query, slot)
    tag_string_wo_CLS_SEP = tag_string

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

    tokensForQueries.append(text)
    labelsForQueries.append(tag_string)

    # do not include CLS SEP for comparison
    labelIdsForQueries.append(slots_label_set.get_aligned_label_ids_from_aligned_label(
        map(str.lower,tag_string_wo_CLS_SEP.split())
    ))

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


'''
print('output training and test file');
#output_file = 'train_bert_email_slot.tsv' if isTrain == True else 'dev_bert_email_slot.tsv'
output_file = '..\\azureml_data\\atis_train.tsv_after_my_preprocessing.tsv'
with codecs.open(output_file, 'w', 'utf-8') as fout:
    for i, (query, labelsforQuery) in enumerate(zip(tokensForQueries, labelsForQueries)):
        fout.write(query.strip() + '\t' + labelsforQuery.strip() + '\r\n');
'''

output_file = '..\\azureml_data\\after_my_preprocessingOriginalSlotIdsv1.tsv'
with codecs.open(output_file, 'w', 'utf-8') as fout:
    for i, (query, labelIdListForQuery) in enumerate(zip(tokensForQueries, labelIdsForQueries)):
        fout.write(query + '\t' + " ".join([str(x) for x in labelIdListForQuery]) + '\r\n');




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

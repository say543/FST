import codecs;
from collections import Counter;
import mmap;
#import nltk;
#from nltk.tokenize import word_tokenize
import numpy as np;
import os;
import pickle;
import re;
import string;
from tqdm import tqdm;
from transformers import BertTokenizer;
import torch;
import torch.nn as nn;

class Preprocess_Driver_Bert:
    def __init__(self, useIob=False):
        self.cutoff_length = 100;
        self.open_pattern = r'<(\w+)>';
        self.close_pattern = r'</(\w+)>';
        self.useIob = useIob;
        self.slot_dict = { 'O':0, 
                            'file_name':1, 
                            'file_type':2, 
                            "data_source":3, 
                            "contact_name":4, 
                            "to_contact_name":5,
                            "file_keyword":6,
                            "date":7,
                            "time":8,
                            "meeting_starttime":9,
                            "file_action":10,
                            "file_action_context":11,
                            "position_ref":12,
                            "order_ref":13,
                            "file_recency":14,
                            "sharetarget_type":15,
                            "sharetarget_name":16,
                            "file_folder":16,
                            "data_source_name":17,
                            "data_source_type":18,
                            "attachment":19};

        self.intent_dict = {'file_search':0,
                            'file_share':1,
                            'file_download':2,
                            'file_other':3,
                            'file_navigate':4,
                            'cancel':5,
                            'confirm':6,
                            'reject':7,
                            'select_none':8};


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu");
        #self.training_set = '..\\..\\Data\\Email\\Email_Slot_Training.tsv';
        #self.dev_set = '..\\..\\Data\\Email\\Email_Slot_Validation.tsv';


        # whole data set
        #self.training_set = '..\\azureml_data\\files_slot_training.tsv';
        #self.dev_set = '..\\azureml_data\\files_slot_training.tsv';

        # single query
        self.training_set = '..\\azureml_data\\files_slot_training_single.tsv';
        self.dev_set = '..\\azureml_data\\files_slot_training_single.tsv';


        # comment intent
        #self.intent_training_set = '..\\..\\Data\\Email\\Email_Intent_Training.tsv';
        #self.intent_dev_set = '..\\..\\Data\\Email\\Email_Intent_Validation.tsv';
 
        # For TinyBert
        #self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', vocab_file='TinyBert\\vocab.txt');
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased');
        self.slot_list = [];

        for key, value in self.slot_dict.items():
            self.slot_list.append('<' + key + '>');
            self.slot_list.append('</' + key + '>');

        print(self.slot_list);

        self.max_length = 64;

    def isOpenPattern(self, word):
        match = re.match(self.open_pattern, word);
        if not match:
            return (None, False);
        slot = match[1].strip().lower();
        if slot in self.slot_dict:
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
        email_action_pattern = r'<email_action>\s*([^<]+?)\s*</email_action>';
        preprocessed = re.sub(email_action_pattern, '\\1', input);

        # remove email quantifier related
        quantifier_pattern = r'<quantifier>\s*([^<]+?)\s*</quantifier>';
        preprocessed = re.sub(quantifier_pattern, '\\1', preprocessed);

        # open up the slot annotation with space
        pattern = r'<([^>]+?)>\s*([^<]+?)\s*</([^>]+?)>';
        preprocessed = re.sub(pattern, '<\\1> \\2 </\\3>', preprocessed); # <date>friday</date> -> <date> friday </date> for further splitting
        if preprocessed[-1] in string.punctuation and preprocessed[-1] != '>':
            punc = preprocessed[-1];
            preprocessed = preprocessed[:-1] + ' ' + punc;
        return preprocessed;

    def checkQueryValid(self, input):
        # query should not contain any unwanted slot annotations (this happens when the original annotation is wrong)
        for error_item in self.slot_list:
            if error_item in input:
                return False;
        return True;

    def getNumLines(self, file_path):
        fp = open(file_path, "r+")
        buf = mmap.mmap(fp.fileno(), 0)
        lines = 0
        while buf.readline():
            lines += 1
        return lines

    def splitWithBert(self, annotation):
        preSplit = annotation.split();
        annotationArray = [];
        for word in preSplit:
            if any(slot in word for slot in self.slot_list):
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

    def preprocessRawFile(self, data_input, isTrain=True):
        pattern = r'<(?P<name>\w+)>(?P<entity>[^<]+)</(?P=name)>';
        feature_parse = [];

        print('parsing: ', data_input);
        with codecs.open(data_input, 'r', 'utf-8') as fin:
            for line in tqdm(fin, total=self.getNumLines(data_input)):
                line = line.lower().strip();
                if not line:
                    continue;
                array = line.split('\t');
                if len(array) < 5:
                    continue;

                try:
                    annotation_original = array[4].strip().lower();
                    annotation = self.simplePreprocessAnnotation(annotation_original);
                    annotation_array = self.splitWithBert(annotation);
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
                        continue;
                    elif len(word_array) == 1:
                        if all(i in string.punctuation for i in word_array):
                            continue;

                    if len(word_array) > self.cutoff_length:
                        word_array = word_array[:self.cutoff_length];
                        annotation_filtered_array = annotation_filtered_array[:self.cutoff_length];

                    # write input string and tag list in file;
                    word_string = " ".join(word_array);
                    tag_string = self.generateTagString(annotation_filtered_array);

                    if not self.checkQueryValid(word_string):
                        continue;

                    feature_parse.append((word_string, tag_string));
                except:
                    continue;
        
        print('output training and test file');
        #output_file = 'train_bert_email_slot.tsv' if isTrain == True else 'dev_bert_email_slot.tsv'
        output_file = '..\\azureml_data\\train_bert_files_slot.tsv' if isTrain == True else '..\\azureml_data\\dev_bert_files_slot.tsv'
        with codecs.open(output_file, 'w', 'utf-8') as fout:
            for item in feature_parse:
                fout.write(item[0] + '\t' + item[1] + '\r\n');
    

    def preprocessRawFileForIntent(self, data_input, isTrain=True):

        feature_parse = [];
        print('parsing: ', data_input);
        with codecs.open(data_input, 'r', 'utf-8') as fin:
            for line in tqdm(fin, total=self.getNumLines(data_input)):
                line = line.lower().strip();
                if not line:
                    continue;
                array = line.split('\t');
                if len(array) < 4:
                    continue;
                query = array[2].strip().lower();
                word_array = self.tokenizer.tokenize(query);
                word_string = " ".join(word_array);
                intents = array[3];
                feature_parse.append((word_string, intents));
        
        print('output training and test file');
        output_file = 'train_bert_email_intent.tsv' if isTrain == True else 'dev_bert_email_intent.tsv'
        with codecs.open(output_file, 'w', 'utf-8') as fout:
            for item in feature_parse:
                fout.write(item[0] + '\t' + item[1] + '\r\n');


    def padOrTruncate(self, tag_int_list):
        if len(tag_int_list) < self.max_length:
            while len(tag_int_list) < self.max_length:
                tag_int_list.append(0);
        else:
            tag_int_list = tag_int_list[0:self.max_length];
            tag_int_list[-1] = 0;
        return torch.as_tensor(tag_int_list, device=self.device, dtype=torch.long);


    def assignIntentIds(self, intent_list):
        intent_ids = [];
        for intent in intent_list:
            intent = intent.strip();
            if intent not in self.intent_dict:
                return (None, False);
            intent_ids.append(self.intent_dict[intent]);

        labels = torch.as_tensor(intent_ids, device=self.device, dtype=torch.long);
        labels_onehot = nn.functional.one_hot(labels, num_classes=len(self.intent_dict));
        labels_onehot = labels_onehot.sum(dim=0).float();
        return (labels_onehot, True);

    def generateTrainTestData(self, input_file):
        print('loading: ' + input_file);
        input_list = [];

        with codecs.open(input_file, 'r', 'utf-8') as fin:
            for line in tqdm(fin, total=self.getNumLines(input_file)):
                line = line.strip();
                if not line:
                    continue;
                array = line.split('\t');
                if len(array) < 2:
                    continue;
                sentence = array[0].strip();
                tagString = array[1].strip();

                word_list = sentence.split(' ');

                encoded_dict = self.tokenizer.encode_plus(
                        word_list,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = self.max_length,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                        truncation=True);


                word_int_list = encoded_dict['input_ids'][0];           
                attention_mask = encoded_dict['attention_mask'][0];

                if len(word_int_list) == 0:
                    print(line);
                    continue;

                tag_list = tagString.split(' ');
                tag_int_list = [];
                for w in tag_list:
                    tag_int_list.append(self.slot_dict[w]);
                
                tag_int_list.insert(0, 0);
                tag_int_list.append(0);

                tag_int_list = self.padOrTruncate(tag_int_list);

                assert(len(word_int_list) == len(tag_int_list) == len(attention_mask));
                mask = attention_mask == 1;
                assert(len(word_int_list[mask]) == len(tag_int_list[mask]));

                assert(tag_int_list[mask][0] == 0);
                assert(tag_int_list[mask][-1] == 0);

                input_list.append((word_int_list, attention_mask, tag_int_list));

        return input_list;

    def generateTrainTestDataForIntent(self, input_file):
        print('loading: ' + input_file);
        input_list = [];

        with codecs.open(input_file, 'r', 'utf-8') as fin:
            for line in tqdm(fin, total=self.getNumLines(input_file)):
                line = line.strip();
                if not line:
                    continue;
                array = line.split('\t');
                if len(array) < 2:
                    continue;
                sentence = array[0].strip();
                intents = array[1].strip();

                word_list = sentence.split(' ');

                encoded_dict = self.tokenizer.encode_plus(
                        word_list,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = self.max_length,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                        truncation=True);


                word_int_list = encoded_dict['input_ids'][0];           
                attention_mask = encoded_dict['attention_mask'][0];

                if len(word_int_list) == 0:
                    print(line);
                    continue;

                intent_list = intents.split(',');
                intent_ids, intent_valid = self.assignIntentIds(intent_list);
                if intent_ids is None or not intent_valid:
                    continue;
                input_list.append((word_int_list, attention_mask, intent_ids));

        return input_list;
    def main(self):
        self.preprocessRawFile(self.training_set, True);
        self.preprocessRawFile(self.dev_set, False);

        #self.preprocessRawFileForIntent(self.intent_training_set, True);
        #self.preprocessRawFileForIntent(self.intent_dev_set, False);

if __name__ == "__main__":
    driver = Preprocess_Driver_Bert(useIob=False);
    driver.main();
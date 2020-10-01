from collections import defaultdict, Counter
import random
import os
from glob import glob
from pprint import pprint
import re
# tgdm is to show progress turning
from tqdm import tqdm
from collections import namedtuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_augmentation import DataAugmentation
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import codecs;

TEST = False


class Tag(object):

    def __init__(self, tag_name, tag_values):
        ## tag : <>  value inside xml
        ## tag values : a list of tokens eg : <> value</>  , va;ue inside   
        self.tag_name = tag_name
        self.tag_values = tag_values

    def get_random_value(self):
        if not self.tag_values:
            raise Exception(
                'The tag values for tag {} are empty!'.format(self.tag_name))
        return random.choices(self.tag_values, k=1)[0]

    def get_values(self):
        return self.tag_values


class Data(object):

    def __init__(self):
        # positive data for data tag augmentation
        #self.posdata_scale_multiplier = 1000  # multiplier to amplify the positive_dataset
        #self.posdata_scale_multiplier = 500  # multiplier to amplify the positive_dataset
        #self.posdata_scale_multiplier = 400  # multiplier to amplify the positive_dataset
        self.posdata_scale_multiplier = 10  # multiplier to amplify the positive_dataset
        #self.posdata_scale_multiplier = 5  # multiplier to amplify the positive_dataset
        #self.posdata_scale_multiplier = 1  # multiplier to amplify the positive_dataset
        #self.posdata_scale_multiplier = 200  # multiplier to amplify the positive_dataset
        self.overtrigger_suffix_upper_bound = 8 # current range 0-9
        self.overtrigger_suffix_value_upper_bound = 90000
        self.overtrigger_suffix_value_low_bound = 10000
        # probability of selecting from additional tags
        ### ? this should be file keywrod / file name only
        ### https://docs.python.org/3/library/collections.html
        ### The arguments are the name of the new class and a string containing the names of the elements.
        ### create class object and string field by this factory method
        TagSelectionProb = namedtuple('TagSelectionProb', ['random_tag_prob', 'additional_tag_prob', 'orig_tag_prob'])
        self.tag_selection_probabilities = TagSelectionProb(random_tag_prob=0.6, additional_tag_prob=0.3,
                                                            orig_tag_prob=0.1)
        self.additional_filetag_selection_cnt = 0
        self.random_filetag_selection_cnt = 0

        #dictionary of list
        self.tags = {}

        # dictionary of dictionay 
        self.overtrigger_tags={}

        self.filekeylistwithsuffix = []

        self.patterns = {}  # pattern: frequency
        # for duplicate pattern, frequency_offset will store the maximum one
        self.patterns_freq_offset = {}  # pattern: frequency_offset
        self.patterns_domain = {}  # pattern: Domain
        self.patterns_annotated_queries = {}  # pattern: annotated queries
        self.patterns_intent = {}  # pattern: annotated queries

        self.additional_patterns_loaded = False
        self.max_freq = 0
        self.high_freq_patterns = {}
        self.low_freq_patterns = {}

        ## list of list
        ## list[0] : query

        self.positive_data = []
        self.positve_data_intent = []

        ## 'filekeyword', 'fileskeyword', 'filename' being used when generating positive data
        ## source can from *.txt or nltk
        self.keylist = []  # list to accumulate file keywords

        self.negative_data = []
        self.negative_ngrams = defaultdict(int)  # dictionary for negative data lookup for filetags

        self.negative_data_intent = []
        self.negative_data_PreviousTurnDomain = []
        self.negative_data_TaskFrameStatus = []
        self.negative_data_TaskFrameEntityStates = []
        self.negative_data_TaskFrameGUID = []
        self.negative_data_SpeechPeopleDisambiguationGrammarMatches = []
        self.negative_data_ConversationalContext = []

        self._filename_gen_retry_list = []

        ## nltk library
        ## https://www.nltk.org/book/ch02.html
        # https://medium.com/pyladies-taiwan/nltk-%E5%88%9D%E5%AD%B8%E6%8C%87%E5%8D%97-%E4%BA%8C-%E7%94%B1%E5%A4%96%E8%80%8C%E5%85%A7-%E5%BE%9E%E8%AA%9E%E6%96%99%E5%BA%AB%E5%88%B0%E5%AD%97%E8%A9%9E%E6%8B%86%E8%A7%A3-%E4%B8%8A%E6%89%8B%E7%AF%87-e9c632d2b16a
        self.vocab = nltk.corpus.words.words()

        self.da = DataAugmentation()

        # read filetype
        # using uwp one to remove picture file
        self.filetypeIncludeBoost = []
        self.fileboost = []
        
        #with codecs.open('..\\resource\\lexicons\\file_type_domain_boost.txt', 'r', 'utf-8') as fin:
        #with codecs.open('..\\resource\\lexicons\\file_type_domain_boost_UWP.txt', 'r', 'utf-8') as fin:
        with codecs.open('..\\..\\resource\\lexicons\\file_type_domain_boost_UWP.txt', 'r', 'utf-8') as fin:
            for line in fin:
                line = line.strip()
                if line.lower() == 'documents' or line.lower() == 'document' or line.lower() == 'file' or line.lower() == 'files':
                    self.fileboost.append(line)
                #else:
                self.filetypeIncludeBoost.append(line)

        # read order_ref
        '''
        self.orderref = []
        with codecs.open('placeholder_tags_chiecha\\order_ref.txt', 'r', 'utf-8') as fin:
            for line in fin:
                line = line.strip()
                self.orderref.append(line)

        # read filerecency
        self.filerecency = []
        with codecs.open('placeholder_tags_chiecha\\file_recency.txt', 'r', 'utf-8') as fin:
            for line in fin:
                line = line.strip()
                self.filerecency.append(line)
        '''

        # comment this since prefer original slot for evaluation
        self.domain_slot_process = defaultdict(set)
        self.domain_slot_process['FILES'].add('file_keyword')
        self.domain_slot_process['FILES'].add('file_name')
        
        '''
        self.domain_slot_process = defaultdict(set)
        self.domain_slot_process['CALENDAR'].add('title')
        self.domain_slot_process['PEOPLE'].add('peopleattribute')
        self.domain_slot_process['TEAMSMESSAGE'].add('keyword')
        self.domain_slot_process['EMAIL'].add('emailsubject')
        self.domain_slot_process['EMAIL'].add('message')
        self.domain_slot_process['EMAIL'].add('keyword')
        self.domain_slot_process['NOTE'].add('notetext')
        self.domain_slot_process['REMINDER'].add('remindertext')
        self.domain_slot_process['FILES'].add('filekeyword')
        self.domain_slot_process['FILES'].add('filename')
        '''


        # tag mapped logic
        self.tag_originalTag = defaultdict(defaultdict)
        # attachment is only token in query pattern (has pair in annotation), no need to map
        #self.tag_originalTag['EMAILSEARCH']['attachment'] = 'attachment'
        self.tag_originalTag['EMAILSEARCH']['attachment_type'] = 'file_type'
        # message_type do not map, map to dedicated tokens
        #self.tag_originalTag['EMAILSEARCH']['message_type'] = 'keyword'
        self.tag_originalTag['EMAILSEARCH']['contact_name'] = 'contact_name'
        self.tag_originalTag['EMAILSEARCH']['from_contact_name'] = 'contact_name'
        self.tag_originalTag['EMAILSEARCH']['contact_name_to'] = 'to_contact_name'
        self.tag_originalTag['EMAILSEARCH']['email_state'] = 'file_action'
        self.tag_originalTag['EMAILSEARCH']['email_subject '] = 'file_keyword'
        self.tag_originalTag['EMAILSEARCH']['end_date'] = 'date'
        self.tag_originalTag['EMAILSEARCH']['end_time'] = 'time'
        self.tag_originalTag['EMAILSEARCH']['keyword'] = 'file_keyword'
        self.tag_originalTag['EMAILSEARCH']['message_category'] = 'file_keyword'
        # order rer leave it to two possible cases, file_recency or originla tag
        #self.tag_originalTag['EMAILSEARCH']['order_ref'] = 'keyword'
        self.tag_originalTag['EMAILSEARCH']['start_date'] = 'date'
        self.tag_originalTag['EMAILSEARCH']['start_time'] = 'time'
        # without_attachment then cancel it directly
        # emailsearch_other  then cancel it direclty




    def _filter_tag_date(self, tag, values):

        print("{} before filter {}".format(tag, len(values)))

        newvalues = []
        if tag != 'file_type':
            for value in values:
                valuetokensSet = set(value.lower().split())

                insidefiletypeIncludeBoost = False
                for element in self.filetypeIncludeBoost:
                    if element.lower() in valuetokensSet:
                        #print("filter token:{} ".format(value))
                        insidefiletypeIncludeBoost = True

                if insidefiletypeIncludeBoost is False:
                    newvalues.append(value)

        print("{} after filter {}".format(tag, len(newvalues)))

        return values

    def load_tags_data(self, file):
        tag = os.path.basename(file).replace('.txt', '')

        # for debug
        #print("tag {}".format(tag))

        ## remain head / tailing spaces
        with open(file, encoding='utf-8') as f:
            values = [val.strip() for val in f.readlines()]


        # filter tag
        values = self._filter_tag_date(tag, values)

        # for debug 
        print("{} before deduplication {}".format(file, len(values)))

        #dedup and transfer back to list
        values = list(set(values))




        # for debug 
        print("{} after deduplication {}".format(file, len(values)))

        if tag not in self.tags:
            self.tags[tag] = Tag(tag, values)
        else:
            raise Exception('Duplicate tags found for {}'.format(tag))


    def load_overtrigger_tags_data(self, file):
        tag = os.path.basename(file).replace('.txt', '')

        # for debug
        #print("tag {}".format(tag))

        ## remain head / tailing spaces
        with open(file, encoding='utf-8') as f:
            values = [val.strip() for val in f.readlines()]


        # for debug 
        print("overtrigger {} before deduplication {}".format(file, len(values)))

        #dedup and transfer back to list
        values = list(set(values))

        # for debug 
        print("overtrigger {} after deduplication {}".format(file, len(values)))

        # for debug 
        #print("overtrigger {} ".format(len(self.overtrigger_tags)))

        if tag not in self.overtrigger_tags:
            self.overtrigger_tags[tag] = Tag(tag, values)
        else:
            raise Exception('Duplicate overtrigger tags found for {}'.format(tag))

    def _augment_patterns(self):
        similar_patterns_dict = {}
        for pattern, freq in self.patterns.items():
            similar_patterns = self.da.get_similar_patterns(pattern)
            similar_patterns_dict.update({sim_pattern: freq for sim_pattern in similar_patterns})

        self.patterns.update(similar_patterns_dict)

    def _load_additional_positive_patterns(self, filename, default_freq):
        with open(filename, encoding='utf-8') as f:
            '''
            additional_patterns = {pattern.strip(): default_freq for pattern in f.readlines()}
            self.patterns.update(additional_patterns)
            self.additional_patterns_loaded = True
            '''


            pattern_data = [line.strip() for line in f.readlines()]


            #addtional patterns frequency is decided by main extra patterns
            # so igonre its frequency
            # freq sorted by high to lower, so using the first one to record frequency
            #self.max_freq = int(pattern_data[0].split('\t')[1])

            domain = pattern_data[0].split('\t')[2]

            # for debug 
            print("domain {}".format(domain))

            # pattern : freq dict
            ## key : pattern, value : frequency
            #self.patterns = {p.split('\t')[0]: int(
            #    p.split('\t')[1]) for p in pattern_data}
            for p in pattern_data:

                # debug
                #print("pattern: {}".format(p))

                #addtional patterns frequency is decided by main extra patterns
                # since it it decided by int(freq * len(tags) * self.posdata_scale_multiplier)
                # so igonre its frequency and setup as original patterns extracted max_freq
                # in this way, all additinoal patterns are treated as high_frequency patterns                
                self.patterns[(p.split('\t')[0])] = self.max_freq

                self.patterns_domain[(p.split('\t')[0])] = domain

                self.patterns_annotated_queries[(p.split('\t')[0])] = p.split('\t')[3]
                self.patterns_intent[(p.split('\t')[0])] = p.split('\t')[4]

                #store freq offset
                self.patterns_freq_offset[(p.split('\t')[0])] = int(p.split('\t')[5])
                



    def load_patterns(self, all_patterns_files, **kwargs):


        for file in tqdm(all_patterns_files):

            print("Loading patterns from {}".format(file))


            with open(file, encoding='utf-8') as f:
                pattern_data = [line.strip() for line in f.readlines()]




                # might ne empty
                if not pattern_data:
                    raise Exception('No positive patterns loaded!')

                #consider loading more patterns files

                # freq sorted by high to lower, so using the first one to record frequency
                #self.max_freq = int(pattern_data[0].split('\t')[1])

                #consider loading more patterns files
                self.max_freq = max(self.max_freq, (int)(pattern_data[0].split('\t')[1]))

                domain = pattern_data[0].split('\t')[2]

                # for debug 
                print("domain {}".format(domain))

                # pattern : freq dict
                ## key : pattern, value : frequency
                #self.patterns = {p.split('\t')[0]: int(
                #    p.split('\t')[1]) for p in pattern_data}
                for p in pattern_data:

                    # debug
                    print("pattern: {}".format(p))

                    self.patterns[(p.split('\t')[0])] = int(p.split('\t')[1])

                    self.patterns_domain[(p.split('\t')[0])] = domain

                    self.patterns_annotated_queries[(p.split('\t')[0])] = p.split('\t')[3]
                    self.patterns_intent[(p.split('\t')[0])] = p.split('\t')[4]


                    # store freq offset
                    self.patterns_freq_offset[(p.split('\t')[0])] = int(p.split('\t')[5])

                    # for debug
                    print("patterns num = {}".format(len(self.patterns)))

                # for debug 
                #print("patterns num = {}".format(len(self.patterns)))
                

    '''
    def load_positive_patterns(self, file, **kwargs):
        print("Loading patterns from {}".format(file))
        with open(file, encoding='utf-8') as f:
            pattern_data = [line.strip() for line in f.readlines()]

            if not pattern_data:
                raise Exception('No positive patterns loaded!')

            # assuming the pattern data is sorted by frequency
            self.max_freq = int(pattern_data[0].split('\t')[1])

            # pattern : freq dict
            ## key : pattern, value : frequency
            self.patterns = {p.split('\t')[0]: int(
                p.split('\t')[1]) for p in pattern_data}


        ## also loading addtional_patterns.txt but this is not passed as an argument
        ## ? better to have its own functionality 
        ## change to my own source
        ## move to another new function    
        #print("Loading additional patterns.")
        ##self._load_additional_positive_patterns('additional_patterns.txt', self.max_freq)
        #self._load_additional_positive_patterns('additional_patterns_chiecha.txt', self.max_freq)
        ## self._augment_patterns()
    '''
    def load_addtional_patterns(self, file):
        print("Loading additional patterns.with {}".format(
            self.max_freq))
        #self._load_additional_positive_patterns('additional_patterns.txt', self.max_freq)
        self._load_additional_positive_patterns(file, self.max_freq)
        # self._augment_patterns()
        # 



    def _attachment_patterns_file_acton_refinement(self, query, annotation):



            new_annotation = annotation

            # ignore lowercase / upper case at first
            verbs = set([
                    #--- no present in data
                    # "downloaded",
                    # "worked",
                    # "created",
                    # "saved",
                    # "made",
                    # "edited",
                    # "took",
                    # "uploaded",
                    # "working",
                    #--- no present in data
                     "shared",
                    #--- no present in data
                    # "wrote",
                    # "added",
                    # "used",
                    # "using",
                    # "composed",
                    # "opened",
                    # "composing",
                    # "morning",
                    # "walked",
                    # "edited",
                    # "updated",
                    # "writing",
                    # "doing",
                    # "did",
                    # "looking",
                    # "looked",
                    # "reviewed",
                    #--- no present in data
                    #--- not supposed to be file_action
                    #"titled",
                    #"called",
                    #--- no present in data
                     "marked",
                    #--- no present in data
                     'sent'
                     ])

            # only replace annotation, do not touch original query
            for verb in verbs:
                # verb space (start with)
                # replace with the first occurence

                if query.startswith(verb +' '): 
                    new_annotation = "<file_action> "+verb+" </file_action>" + ' ' + new_annotation[len(verb)+1:]



                # verb space (end with)
                # replace with the first occurence

                if query.endswith(' '+verb): 
                    new_annotation = new_annotation[0:len(new_annotation)-len(verb)-1] +' '+"<file_action> "+verb+" </file_action>"

                
                # return multiple occurence
                if query.find(' '+ verb +' ') != -1:
                    new_annotation = new_annotation.replace(' '+ verb +' ', ' '+"<file_action> "+verb+" </file_action>"+' ')
                

                # for debug
                #print("file_action populuted annotation from: {} to: {}".format(
                #    annotation, new_annotation))


            return query, new_annotation

    # even appliyng this function no all cases being covered. 
    def _attachment_patterns_date_time_map_annotation(self, query, annotation):
        new_annotation = annotation

        xmlpairs = re.findall("(<.*?>.*?<\/.*?>)", annotation)
        for xmlpair in xmlpairs:
            # extra type and value for xml tag
            xmlTypeEndInd = xmlpair.find(">")

            xmlType = xmlpair[1:xmlTypeEndInd]

            xmlValue = xmlpair.replace("<"+xmlType+">", "")
            xmlValue = xmlValue.replace("</"+xmlType+">", "")
            xmlValue = xmlValue.strip()

            if xmlType.lower() == 'start_date':
                new_annotation = new_annotation.replace("<"+xmlType+">", '<{}>'.format('date'))
                new_annotation = new_annotation.replace("</"+xmlType+">", '</{}>'.format('date'))

            if xmlType.lower() == 'end_date':
                new_annotation = new_annotation.replace("<"+xmlType+">", '<{}>'.format('date'))
                new_annotation = new_annotation.replace("</"+xmlType+">", '</{}>'.format('date'))

            if xmlType.lower() == 'start_time':
                new_annotation = new_annotation.replace("<"+xmlType+">", '<{}>'.format('time'))
                new_annotation = new_annotation.replace("</"+xmlType+">", '</{}>'.format('time'))

            if xmlType.lower() == 'end_time':
                new_annotation = new_annotation.replace("<"+xmlType+">", '<{}>'.format('time'))
                new_annotation = new_annotation.replace("</"+xmlType+">", '</{}>'.format('time'))

        return query, new_annotation

    def append_attachment_patterns(self, file, newdomain, newintent):
        print("append additional attachment patterns.with {}".format(
            self.max_freq))
        with open(file, encoding='utf-8') as f:


            pattern_data = [line.strip() for line in f.readlines()]


            #addtional patterns frequency is decided by main extra patterns
            # so igonre its frequency
            # freq sorted by high to lower, so using the first one to record frequency
            #self.max_freq = int(pattern_data[0].split('\t')[1])

            for p in pattern_data:

                query = p.split('\t')[0]
                domain = p.split('\t')[2]
                annotation = p.split('\t')[3]
                intent = p.split('\t')[4]
                freq_offset = p.split('\t')[5]
                
                #populuate file_action to annotation know
                query, annotation = self._attachment_patterns_file_acton_refinement(query, annotation)
                query, annotation = self._attachment_patterns_date_time_map_annotation(query, annotation)

                # for debug
                #print("mapped domain from domain {} to {}".format(domain, newdomain))
                #print("mapped intent from intent {} to {}".format(intent, newintent))

                # finding tags in a query
                tags = re.findall(r'<(.*?)>', (p.split('\t')[0]))
                for tag in tags:
                    if tag not in self.tag_originalTag[domain]:
                        if tag == 'attachment':
                            continue
                        elif tag == 'message_type':
                            inRangeIndex = random.randint(0, len(self.fileboost)-1)

                            random_tag = self.fileboost[inRangeIndex]
                            query = query.replace("<{}>".format(tag), random_tag)
                            # file boost xml is removed here
                            annotation = annotation.replace("<{}>".format(tag), 
                            " {} ".format(random_tag))
                        elif tag == 'order_ref':
                            inRangeIndex = random.randint(0, 2)
                            # 0.5 prob being replaced with file_recency
                            if inRangeIndex == 1: 
                                #query = query.replace("<{}>".format(tag, 'file_recency'))
                                query = query.replace("<{}>".format(tag), "<{}>".format('file_recency'))
                                # file boost xml is removed here
                                annotation = annotation.replace("<{}>".format(tag), 
                                    "<{}>".format('file_recency'))
                        elif tag == 'contact_name_from':
                            #query = query.replace("<{}>".format(tag, 'contact_name'))
                            query = query.replace("<{}>".format(tag), "<{}>".format('contact_name'))
                            # file boost xml is removed here
                            annotation = annotation.replace("<{}>".format(tag), 
                                "<{}>".format('contact_name'))
                        elif tag == 'contact_name_to':
                            #query = query.replace("<{}>".format(tag, 'to_contact_name'))
                            query = query.replace("<{}>".format(tag), "<{}>".format('to_contact_name'))
                            # file boost xml is removed here
                            annotation = annotation.replace("<{}>".format(tag), 
                                "<{}>".format('to_contact_name'))


                        # no exist in pattern                      
                        #else if tag == 'without_attachment' or  tag == 'emailsearch_other':
                        else:
                            raise Exception('Unknown tag {} found in the appended query pattern {}'.format(tag, p))


                        '''
                        elif tag == 'keyword':
                            query = query.replace("<{}>".format(tag, 'file_keyword'))
                            # file boost xml is removed here
                            annotation = annotation.replace("<{}>".format(tag), 
                                "<{}>".format('file_keyword'))
                        '''

                    else:
                        query = query.replace("<{}>".format(tag), "<{}>".format(self.tag_originalTag[domain][tag]))
                        # file boost xml is removed here
                        annotation = annotation.replace("<{}>".format(tag), 
                            "<{}>".format(self.tag_originalTag[domain][tag]))

                # debug
                print("mapped query pattern: {}".format(query))

                #addtional patterns frequency is decided by main extra patterns
                # since it it decided by int(freq * len(tags) * self.posdata_scale_multiplier)
                # so igonre its frequency and setup as original patterns extracted max_freq
                # in this way, all additinoal patterns are treated as high_frequency patterns                
                self.patterns[query] = self.max_freq

                self.patterns_domain[query] = newdomain

                self.patterns_annotated_queries[query] = annotation
                self.patterns_intent[query] = newintent

                #store freq offset
                self.patterns_freq_offset[query] = int(freq_offset)

        
    def split_patterns(self, pattern_selection_threshold=0.05):
        """ splits the positive patterns into high freq and low freq patterns

        Args:
            pattern_selection_threshold (float, optional): Threshold for the split.
        """

        if self.max_freq == 0:
            raise Exception('Cannot perform split with no data')

        # plot histogram for freq distribution
        # sns.distplot(list(self.patterns.values()))
        # plt.show()

        ## ? not sure how this frequency work
        ## it seems like originla pattern 's freq still matter
        # reason we split between high freq and low freq patterns initially 
        # is bacause of such behaviour:
        # Notice annotation in CSDS repo in teams data is brokem for such low freq patterns
        split_freq = int(self.max_freq * pattern_selection_threshold)


        print("Generating positive data from patterns.")
        print("split_freq: {}".format(
            split_freq))

        ## for pattern frequency >= split_freq, cap it with 
        for pattern, freq in self.patterns.items():

            # for debug
            #print("pattern: {}, freq: {} ".format(
            #pattern, freq))

            if freq >= split_freq:
                self.high_freq_patterns[pattern] = freq
            else:
                self.low_freq_patterns[pattern] = freq

        pprint(len(self.high_freq_patterns))
        pprint("-" * 20)
        pprint(len(self.low_freq_patterns))

    '''
    def split_positive_patterns(self, pattern_selection_threshold=0.05):
        """ splits the positive patterns into high freq and low freq patterns

        Args:
            pattern_selection_threshold (float, optional): Threshold for the split.
        """

        if self.max_freq == 0:
            raise Exception('Cannot perform split with no data')

        # plot histogram for freq distribution
        # sns.distplot(list(self.patterns.values()))
        # plt.show()

        ## ? not sure how this frequency work
        ## it seems like originla pattern 's freq still matter
        split_freq = int(self.max_freq * pattern_selection_threshold)


        print("Generating positive data from patterns.")
        print("split_freq: {}".format(
            split_freq))

        ## for pattern frequency >= split_freq, cap it with 
        for pattern, freq in self.patterns.items():

            # for debug
            print("pattern: {}, freq: {} ".format(
            pattern, freq))

            if freq >= split_freq:
                self.high_freq_patterns[pattern] = freq
            else:
                self.low_freq_patterns[pattern] = freq

        pprint(len(self.high_freq_patterns))
        pprint("-" * 20)
        pprint(len(self.low_freq_patterns))
    '''

    def _add_tagvalue_to_keylist(self, tag_value):
        self.keylist.append(tag_value)

    def _filetag_selection(self):
        """Tells from where the selection should be for file tags

        Returns:
            (bool): Returns the selection source
        """
        selection = \
            random.choices(["Random Tag", "Additional Tag", "Original Tag"], k=1,
                           weights=list(self.tag_selection_probabilities))[0]
        return selection



    def _get_new_suffix_for_filename_with_prob(self, filename, variationSource):

        #for debug
        #print("random_tag prepared for suffix {}".format(filename))

        generated_filename = filename.copy()

        '''
        overtrigger_tags_set = set(self.overtrigger_tags['file_keyword'].get_values())
        for i in range(len(filename)):
            if filename[i].lower() in overtrigger_tags_set:


                # with prob 0.7 that if will append new suffix
                rand_prb = random.randint(0, 9)
                if rand_prb <=self.overtrigger_suffix_upper_bound:
                    filename[i] += str(random.randint(self.overtrigger_suffix_value_low_bound, 
                        self.overtrigger_suffix_value_upper_bound))
                    # for debug
                    #print("new suffix: {}, freq: {} ".format(filename[i], " ".join(filename)))

                    self.filekeylistwithsuffix.append(filename[i])

        # for debug
        #print("new filename len:{} with suffux: {}".format(len(filename),  " ".join(filename)))
        #return " ".join(filename)
        '''

        generated_filename = filename.copy()
        overtrigger_tags_set = set(self.overtrigger_tags['file_keyword'].get_values())
        for i in range(len(generated_filename)):
            if filename[i].lower() in overtrigger_tags_set:


                # with prob overtrigger_suffix_upper_bound+1 that if will append new suffix
                rand_prb = random.randint(0, 9)

                # for debug
                #if (filename[i].lower() == 'mswAprilMayQAS')
                #    print("rand_prb:{}".format(len(filename)))
                #    raise Exception('Cannot perform split with no data')


                if rand_prb <=self.overtrigger_suffix_upper_bound:
                    generated_filename[i] += str(random.randint(self.overtrigger_suffix_value_low_bound, 
                        self.overtrigger_suffix_value_upper_bound))
                    # for debug
                    #print("new suffix: {}, freq: {} ".format(generated_filename[i], " ".join(generated_filename)))

                    self.filekeylistwithsuffix.append(variationSource + '\t'+ 
                    " ".join(filename)+'\t'+
                    generated_filename[i])


        return " ".join(generated_filename)

    def _get_random_generated_filetag(self, variationSource):
        # This fn checks the effects of filename/keyword selection
        random_filename = []
        num_words = random.randint(1, 4)

        # for slot it does not have negative data so ignore it at first
        #if not self.negative_ngrams:
        #    raise Exception("Negative patterns not loaded")

        attempt = 0
        max_attempts = 10
        while attempt < max_attempts:

            ## generate random words from self.vocad(nltk corpus)
            for _ in range(num_words):
                random_filename.extend(random.choices(self.vocab, k=1))

            # add suffix according to overtrigger tag
            generated_filename = self._get_new_suffix_for_filename_with_prob(random_filename, variationSource)
            '''
            overtrigger_tags_set = set(self.overtrigger_tags['file_keyword'].get_values())
            for i in range(len(random_filename)):
                if random_filename[i].lower() in overtrigger_tags_set:
                    # with prob 0.7 that if will append new suffix
                    rand_prb = random.randint(0, 9)
                    if rand_prb <=self.overtrigger_suffix_upper_bound:
                        random_filename[i] += str(random.randint(self.overtrigger_suffix_low_bound, self.overtrigger_suffix_upper_bound))
                        # for debug
                        #print("new suffix: {}, freq: {} ".format(random_filename[i], " ".join(random_filename)))

                        self.filekeylistwithsuffix.append(random_filename[i])
                        raise Exception('inside here')
            generated_filename = " ".join(random_filename)
            '''

            ## if generated_filename inside negative n_grams, then igonre it and retry given max attempts
            ##? but once attempt reachs limit it it will return directly 
            if self.negative_ngrams[generated_filename] > 0:
                attempt += 1
                self._filename_gen_retry_list.append(generated_filename)
            else:
                return generated_filename

        return generated_filename

    def _get_randomtag_queries(self, pattern, freq):
        """ for given pattern, generates freq number of queries randomly by replacing tags randomly from taglist

        Args:
            pattern (string): query pattern
            freq (int): number of query pattern to be generated

        Raises:
            Exception: if a query with unknown tag if found

        Returns:
            queries (list of strings): list of generated queries 
        """
        # Salutations
        # ignore salutations
        
        salutations = [
            "",
            "cortana ",
            "hey cortana ",
            "ok cortana ",
            "can you ",
            "please ",
            "hello, ",
            "cortana, can you ",
            "hey cortana, can you ",
            "ok cortana, can you ",
            "cortana, please ",
            "hey cortana, please ",
            "ok cortana, please ",
            "hello cortana, "
        ]



        # finding tags in a query
        tags = re.findall(r'<(.*?)>', pattern)


        #if len(tags) == 0:
        #    freq = 1  # self.posdata_scale_multiplier
        #else:
        #    freq = int(freq * len(tags) * self.posdata_scale_multiplier)

        # add freq offset 
        # at least repeat one time
        if len(tags) == 0:
            freq = 1  # self.posdata_scale_multiplier
        else:
            # for deubg
            #print("pattern: {}, patterns_freq_offset: {}".format(pattern, self.patterns_freq_offset[pattern]))

            freq = int(freq * len(tags) * (max(1, self.posdata_scale_multiplier + self.patterns_freq_offset[pattern])))



        ## for debug
        ## for high frequency patterns : 
        ## posdata_scale_multiplier(defualt 1000) * len(tags) * high_freq_multiplier(9)
        ## for lower frequency patterns: 
        ## posdata_scale_multiplier(defualt 1000) * len(tags) * low_freq_multiplier(1) 

        # but finally there is a place to remove duplicate queries       
        print("pattern: {}, generate_freq: {}".format(
            pattern, freq))

        queries = []
        queries_domain = {}  # queries: Domain
        queries_annotated_queries = {}  # queries: annotated queries
        queries_intent = {}  # queries: intent
        
        #for _ in range(freq):
        for iter in range(freq):

            ## use pattern as originla query
            query = pattern

            domain = self.patterns_domain[pattern]
            annotation = self.patterns_annotated_queries[pattern]
            intent =  self.patterns_intent[pattern]

            for tag in tags:


                if tag not in self.tags:
                    raise Exception(
                        'Unknown tag {} found in the query pattern {}'.format(tag, pattern))
                
#                if tag not in self.domain_slot_process[domain]:
#                    raise Exception(
#                        'Unknown tag {} found in the query pattern {}'.format(tag, pattern))

#                tagWoUnderSocre = tag.replace("_".format(tag), "")
#
#                if tagWoUnderSocre not in self.tags:
#                    raise Exception(
#                        'Unknown tag {} found in the query pattern {}'.format(tagWoUnderSocre, pattern))

                else:

                    # share <filekeyword> -> share OKR
                    # do not use probably to replace
                    # use keyphrases to replace



                    ## _filetag_selection can choose from Additional Tag , Randpm Tag, Original tag
                    filetag_src = self._filetag_selection()

                    ## randomly generating for three slots 'filekeyword', 'fileskeyword', 'filename'
                    ## filekeyword, fileskeyword will use UserFileNamesKeyPhrase
                    ## filename will use UserFileNames
                    ## ? having fileskeyword should be data having some issues
                    if tag in ['file_keyword', 'files_keyword'] and filetag_src == 'Additional Tag':
                        random_tag = self.tags['additionalfilenameskeyphrases'].get_random_value()

                        random_tag = self._get_new_suffix_for_filename_with_prob(random_tag.split(), 'additionalfilenameskeyphrases_tag')


                        self.additional_filetag_selection_cnt += 1
                    elif tag in ['file_name'] and filetag_src == 'Additional Tag':
                        random_tag = self.tags['additionalfilenames'].get_random_value()

                        random_tag = self._get_new_suffix_for_filename_with_prob(random_tag.split(), 'additionalfilenames_tag')


                        self.additional_filetag_selection_cnt += 1
                    elif tag in ['file_keyword', 'files_keyword', 'file_name'] and filetag_src == 'Random Tag':
                        random_tag = self._get_random_generated_filetag("Random_Tag")

                        self.random_filetag_selection_cnt += 1
                    elif tag in ['contact_name'] or tag in ['to_contact_name'] :
                        random_tag = self.tags['combine_lexicon'].get_random_value()
                    elif tag in ['file_recency']:
                        random_tag = self.tags['file_recency'].get_random_value()
                    elif tag in ['order_ref']:
                        random_tag = self.tags['order_ref'].get_random_value()


                    else:
                        # this key will key from file_keyword and it needs to add suffix
                        random_tag = self.tags[tag].get_random_value()

                        # for debug
                        #print("random_tag prepared for suffix {}".format(random_tag.split()))
                        #print("overtrigger: {}".format(self.overtrigger_tags))

                        # default using space to split
                        random_tag = self._get_new_suffix_for_filename_with_prob(random_tag.split(), 'file_keyword_tag')

                        # add suffix according to overtrigger tag
                        '''
                        random_tag_tokens = random_tag.split(" ")
                        overtrigger_tags_set = set(self.overtrigger_tags['file_keyword'].get_values())




                        for i in range(len(random_tag_tokens)):
                            # for debug
                           # print("new suffix: {}".format(random_tag_tokens[i]))

                            if random_tag_tokens[i].lower() in overtrigger_tags_set:
                                # with prob 0.7 that if will append new suffix
                                rand_prb = random.randint(0, 9)
                                
                                # for debug
                                #print("rand_prb: {}".format(rand_prb))
                                if rand_prb <= self.overtrigger_suffix_upper_bound:
                                    
                                    # for debug
                                    #print("suffix: {}".format(str(random.randint(1900, 2100))))

                                    random_tag_tokens[i]  += str(random.randint(self.overtrigger_suffix_low_bound, self.overtrigger_suffix_upper_bound))
                                    #print("new suffix: {}, freq: {} ".format(random_tag_tokens[i], " ".join(random_tag_tokens)))
                                    self.filekeylistwithsuffix.append(random_tag_tokens[i])
                        '''
                        

                    if tag in ['file_keyword', 'files_keyword', 'file_name']:
                        # for deubg
                        #print("new random tag: {}".format(random_tag))

                        self._add_tagvalue_to_keylist(random_tag)

                    '''
                    ## _filetag_selection can choose from Additional Tag , Randpm Tag, Original tag
                    #filetag_src = self._filetag_selection()
                    random_tag = self.tags['additionalfilenames'].get_random_value()
                    self._add_tagvalue_to_keylist(random_tag)
                    '''

                    query = query.replace("<{}>".format(tag), random_tag)
                    #annotation = annotation.replace("<{}>".format(tag), "<{}>".format(self.tag_originalTag[domain][tag]))

                    annotation = annotation.replace("<{}>".format(tag), 
                        "<{}> {} </{}>".format(tag, random_tag, tag))

                    # append salutions randomly
                    # no saluations
                    #prefix = random.choices(salutations, k=1)[0]
                    #query = prefix + query
                    #annotation = prefix + annotation


            ##for debug
            #print("query: {}".format(query))
            queries.append(query)

            queries_domain[query] = domain
            queries_annotated_queries[query] = annotation
            queries_intent[query] = intent


            ## for debug
            #print("query: {}".format(query))
            #print("intent: {}".format(intent))
            #print("annotation: {}".format(annotation))


        # for debug
        #print("len: {}".format(len(queries)))


        return queries, queries_domain, queries_annotated_queries, queries_intent

    '''
    def _get_randomtag_queries(self, pattern, freq):
        """ for given pattern, generates freq number of queries randomly by replacing tags randomly from taglist

        Args:
            pattern (string): query pattern
            freq (int): number of query pattern to be generated

        Raises:
            Exception: if a query with unknown tag if found

        Returns:
            queries (list of strings): list of generated queries 
        """
        # Salutations
        salutations = [
            "",
            "cortana ",
            "hey cortana ",
            "ok cortana ",
            "can you ",
            "please ",
            "hello, ",
            "cortana, can you ",
            "hey cortana, can you ",
            "ok cortana, can you ",
            "cortana, please ",
            "hey cortana, please ",
            "ok cortana, please ",
            "hello cortana, "
        ]

        # finding tags in a query
        tags = re.findall(r'<(.*?)>', pattern)

        ##if not tags, do not apply freq for augmentation
        
        if len(tags) == 0:
            freq = 1  # self.posdata_scale_multiplier
        else:
            freq = int(freq * len(tags) * self.posdata_scale_multiplier)

        ## for debug
        ## for high frequency patterns : 
        ## posdata_scale_multiplier(defualt 1000) * len(tags) * high_freq_multiplier(9)
        ## for lower frequency patterns: 
        ## posdata_scale_multiplier(defualt 1000) * len(tags) * low_freq_multiplier(1) 

        # but finally there is a place to remove duplicate queries       
        print("pattern: {}, generate_freq: {}".format(
            pattern, freq))

        queries = []
        #for _ in range(freq):
        for iter in range(freq):

            ## use pattern as originla query
            query = pattern

            for tag in tags:
                if tag not in self.tags:
                    raise Exception(
                        'Unknown tag {} found in the query pattern {}'.format(tag, pattern))
                else:
                    # share <filekeyword> -> share OKR

                    ## _filetag_selection can choose from Additional Tag , Randpm Tag, Original tag
                    filetag_src = self._filetag_selection()

                    ## randomly generating for three slots 'filekeyword', 'fileskeyword', 'filename'
                    ## filekeyword, fileskeyword will use UserFileNamesKeyPhrase
                    ## filename will use UserFileNames
                    ## ? having fileskeyword should be data having some issues
                    if tag in ['filekeyword', 'fileskeyword'] and filetag_src == 'Additional Tag':
                        random_tag = self.tags['additionalfilenameskeyphrases'].get_random_value()
                        self.additional_filetag_selection_cnt += 1
                    elif tag in ['filename'] and filetag_src == 'Additional Tag':
                        random_tag = self.tags['additionalfilenames'].get_random_value()
                        self.additional_filetag_selection_cnt += 1
                    elif tag in ['filekeyword', 'fileskeyword', 'filename'] and filetag_src == 'Random Tag':
                        random_tag = self._get_random_generated_filetag()
                        self.random_filetag_selection_cnt += 1
                    else:
                        random_tag = self.tags[tag].get_random_value()

                    if tag in ['filekeyword', 'fileskeyword', 'filename']:
                        self._add_tagvalue_to_keylist(random_tag)

                    ##for debug
                    #print("random_tag: {}, iteration {}".format(
            #random_tag,iter))

                    query = query.replace("<{}>".format(tag), random_tag)

                    # append salutions randomly
                    prefix = random.choices(salutations, k=1)[0]
                    query = prefix + query



            ##for debug
            #print("query: {}".format(query))
            #queries.append(query)

        #print("len: {}".format(len(queries)))


        return queries
    '''


    def _get_queries_domains_annotated_queries_intent(self, pattern_dict, freq_multiplier):
        queries_domains_annotated_queries_intent = []

        # for each pattern type, replace tags with random tag values
        for pattern, _ in tqdm(pattern_dict.items()):  # Removed freq


            queries, queries_domain, queries_annotated_queries, queries_intent = self._get_randomtag_queries(
                #pattern, freq*freq_multiplier)) # Removed dataset freq to give equal importance
                pattern, freq_multiplier) # Removed dataset freq to give equal importance

            for query in queries:
                queries_domains_annotated_queries_intent.append("{}\t{}\t{}\t{}".format(query, queries_domain[query], queries_annotated_queries[query], queries_intent[query]))

        ##for debug
        print("number ofqueries_domains_annotated_queries_intent: {}".format(
            len(queries_domains_annotated_queries_intent)))


        return queries_domains_annotated_queries_intent



    '''
    def _get_queries(self, pattern_dict, freq_multiplier):
        queries = []

        # for each pattern type, replace tags with random tag values
        for pattern, _ in tqdm(pattern_dict.items()):  # Removed freq
            queries.extend(self._get_randomtag_queries(
                # pattern, freq*freq_multiplier)) # Removed dataset freq to give equal importance
                pattern, freq_multiplier))

        ##for debug
        print("number of queries: {}".format(
            len(queries)))

        return queries
    '''

    #def get_data(self, high_freq_multiplier=9, low_freq_multiplier=1):
    #def get_data(self, high_freq_multiplier=4, low_freq_multiplier=1):
    #def get_data(self, high_freq_multiplier=2, low_freq_multiplier=1):
    def get_data(self, high_freq_multiplier=1, low_freq_multiplier=1):
        print("Generating data from patterns.")

        ## positive_data : list of list
        ## 1 to 1 correspondingly
        ## list[0]: queries lists
        ## list[1]: pattern lists
        ## list[2]: frequency multiplier list
        ## ? rename positive data to data in the future since includes positive data and negative data

        ## dataset's frequency is removed
        ## but here high_freq / low freq still matters 
        self.positive_data.extend(self._get_queries_domains_annotated_queries_intent(
            self.high_freq_patterns, high_freq_multiplier))
        self.positive_data.extend(self._get_queries_domains_annotated_queries_intent(
            self.low_freq_patterns, low_freq_multiplier))

        ## do deduplication for positive_data : list of list
        ## here perform deduplication for all positive generated data
        self.positive_data = set(self.positive_data)

        ## keylist stores what filekeyword / fileskeyword/ filesname selected through positive data generating
        ## can be from *.txt or nltk
        ## here do deduplication
        self.keylist = set(self.keylist)


    '''
    def get_positive_data(self, high_freq_multiplier=9, low_freq_multiplier=1):
        print("Generating positive data from patterns.")

        ## positive_data : list of list
        ## 1 to 1 correspondingly
        ## list[0]: queries lists
        ## list[1]: pattern lists
        ## list[2]: frequency multiplier list

        ## dataset's frequency is removed
        ## but here high_freq / low freq still matters 
        self.positive_data.extend(self._get_queries(
            self.high_freq_patterns, high_freq_multiplier))
        self.positive_data.extend(self._get_queries(
            self.low_freq_patterns, low_freq_multiplier))

        ## do deduplication for positive_data : list of list
        ## here perform deduplication for all positive generated data
        self.positive_data = set(self.positive_data)

        ## keylist stores what filekeyword / fileskeyword/ filesname selected through positive data generating
        ## can be from *.txt or nltk
        ## here do deduplication
        self.keylist = set(self.keylist)
    '''

    def _clean_query(self, query):
        ## preprocessing queries to get ngram feature

        exclude_words = []
        ## remove special string.punctuation characters
        query = ''.join(ch for ch in query if ch not in set(string.punctuation))

        ## here exclude_words are empty so might be useless?
        query_filtered = [w for w in word_tokenize(query.lower()) if w not in exclude_words]

        ## extra grams from 1 to 5
        for i in range(1, 5):

            ## nltk ngram feature
            ith_gram = ngrams(query_filtered, i)

            ## record each gram's frequency
            for gram in ith_gram:
                self.negative_ngrams[' '.join(gram)] += 1
        return query  # ' '.join(query_filtered)

    def _get_random_negative_samples(self, num_samples):
        if num_samples > len(self.negative_data):
            return self.negative_data
        return random.choices(self.negative_data, k=num_samples)

    def _preprocess_negative_data(self):
        # _clean_query : preprocessing queries to get ngram feature
        # v1
        # here dedup negative data
        #self.negative_data = list(set([self._clean_query(query) for query in tqdm(self.negative_data)]))
        # v2
        # do not dedup to prevent from overtriggeing
        #self.negative_data = [self._clean_query(query) for query in tqdm(self.negative_data)]
        # v3
        # do not dedup to prevent from overtriggeing
        # preprocesing ngram but negative_data not being updated
        [self._clean_query(query) for query in tqdm(self.negative_data)]

    def _filter_negative_data(self):
        self.negative_data = self._get_random_negative_samples(int(len(self.negative_data) * 0.7))
        print("Filtering negative data to remove overlap with file keyword list.")
        original_corpus_len = len(self.negative_data)
        random_filekeys_sample = random.choices(list(self.keylist), k=70000)

        for filekey in tqdm(random_filekeys_sample):
            self.negative_data = [query for query in self.negative_data if filekey not in query]

        print("{} queries were filtered".format(original_corpus_len - len(self.negative_data)))

    def _load_additional_negative_data(self, filename='additional_neg_data_in_query_form.txt'):

        # add \t split to extra 'extra negative data'
        # to make sure 'extra negative data' consistent with negative data format

        '''
        with open(filename, encoding='utf-8') as f:
            neg_data = [line.split('\t')[2]
                                  for line in f.readlines()[1:]]
        with open(filename, encoding='utf-8') as f:
            negative_data_intent = [line.split('\t')[3]
                                  for line in f.readlines()[1:]]

        with open(filename, encoding='utf-8') as f:
            negative_data_PreviousTurnDomain = [line.split('\t')[4]
                                  for line in f.readlines()[1:]]

        with open(filename, encoding='utf-8') as f:
            negative_data_TaskFrameStatus = [line.split('\t')[5]
                                  for line in f.readlines()[1:]]
        with open(filename, encoding='utf-8') as f:
            negative_data_TaskFrameEntityStates = [line.split('\t')[6]
                                  for line in f.readlines()[1:]]
        with open(filename, encoding='utf-8') as f:
            negative_data_TaskFrameGUID = [line.split('\t')[7]
                                  for line in f.readlines()[1:]]

        with open(filename, encoding='utf-8') as f:
            negative_data_SpeechPeopleDisambiguationGrammarMatches = [line.split('\t')[8]
                                  for line in f.readlines()[1:]]

        with open(filename, encoding='utf-8') as f:
            negative_data_ConversationalContext = [line.split('\t')[9]
                                  for line in f.readlines()[1:]]
        '''


        with open(filename, encoding='utf-8') as f:
            neg_data = [line.rstrip('\n').split('\t')[2]
                                  for line in f.readlines()[1:]]
        with open(filename, encoding='utf-8') as f:
            negative_data_intent = [line.rstrip('\n').split('\t')[3]
                                  for line in f.readlines()[1:]]

        with open(filename, encoding='utf-8') as f:
            negative_data_PreviousTurnDomain = [line.rstrip('\n').split('\t')[4]
                                  for line in f.readlines()[1:]]

        with open(filename, encoding='utf-8') as f:
            negative_data_TaskFrameStatus = [line.rstrip('\n').split('\t')[5]
                                  for line in f.readlines()[1:]]
        with open(filename, encoding='utf-8') as f:
            negative_data_TaskFrameEntityStates = [line.rstrip('\n').split('\t')[6]
                                  for line in f.readlines()[1:]]
        with open(filename, encoding='utf-8') as f:
            negative_data_TaskFrameGUID = [line.rstrip('\n').split('\t')[7]
                                  for line in f.readlines()[1:]]

        with open(filename, encoding='utf-8') as f:
            negative_data_SpeechPeopleDisambiguationGrammarMatches = [line.rstrip('\n').split('\t')[8]
                                  for line in f.readlines()[1:]]

        with open(filename, encoding='utf-8') as f:
            negative_data_ConversationalContext = [line.rstrip('\n').split('\t')[9]
                                  for line in f.readlines()[1:]]


        '''
        with open(filename, encoding='utf-8') as f:
            neg_data = [line.strip().split('\t')[2]
                                  for line in f.readlines()[1:]]
        with open(filename, encoding='utf-8') as f:
            negative_data_intent = [line.strip().split('\t')[3]
                                  for line in f.readlines()[1:]]

        with open(filename, encoding='utf-8') as f:
            negative_data_PreviousTurnDomain = [line.strip().split('\t')[4]
                                  for line in f.readlines()[1:]]

        with open(filename, encoding='utf-8') as f:
            negative_data_TaskFrameStatus = [line.strip().split('\t')[5]
                                  for line in f.readlines()[1:]]
        with open(filename, encoding='utf-8') as f:
            negative_data_TaskFrameEntityStates = [line.strip().split('\t')[6]
                                  for line in f.readlines()[1:]]
        with open(filename, encoding='utf-8') as f:
            negative_data_TaskFrameGUID = [line.strip().split('\t')[7]
                                  for line in f.readlines()[1:]]

        with open(filename, encoding='utf-8') as f:
            negative_data_SpeechPeopleDisambiguationGrammarMatches = [line.strip().split('\t')[8]
                                  for line in f.readlines()[1:]]

        with open(filename, encoding='utf-8') as f:
            negative_data_ConversationalContext = [line.strip().split('\t')[9]
                                  for line in f.readlines()[1:]]
        '''                          

        #with open(filename, encoding='utf-8') as f:
        #    neg_data = [line.strip() for line in f.readlines()]


        #for debug 
        #print("extra negative data before preprocessing {}".format(neg_data))

        self.negative_data.extend(neg_data)
        self.negative_data_intent.extend(negative_data_intent)
        self.negative_data_PreviousTurnDomain.extend(negative_data_PreviousTurnDomain)
        self.negative_data_TaskFrameStatus.extend(negative_data_TaskFrameStatus)
        self.negative_data_TaskFrameEntityStates.extend(negative_data_TaskFrameEntityStates)
        self.negative_data_TaskFrameGUID.extend(negative_data_TaskFrameGUID)
        self.negative_data_SpeechPeopleDisambiguationGrammarMatches.extend(negative_data_SpeechPeopleDisambiguationGrammarMatches)
        self.negative_data_ConversationalContext.extend(negative_data_ConversationalContext)

    def load_negative_data(self, data_filename, extra_data_filename = None):




        '''
        with open(data_filename, encoding='utf-8') as f:
            self.negative_data = [line.strip().split('\t')[2]
                                  for line in f.readlines()[1:]]
        with open(data_filename, encoding='utf-8') as f:
            self.negative_data_intent = [line.strip().split('\t')[3]
                                  for line in f.readlines()[1:]]
                              
        with open(data_filename, encoding='utf-8') as f:
            self.negative_data_PreviousTurnDomain = [line.strip().split('\t')[4]
                                  for line in f.readlines()[1:]]
        with open(data_filename, encoding='utf-8') as f:
            self.negative_data_TaskFrameStatus = [line.strip().split('\t')[5]
                                  for line in f.readlines()[1:]]

        with open(data_filename, encoding='utf-8') as f:
            self.negative_data_TaskFrameEntityStates = [line.strip().split('\t')[6]
                                  for line in f.readlines()[1:]]
        with open(data_filename, encoding='utf-8') as f:
            self.negative_data_TaskFrameGUID = [line.strip().split('\t')[7]
                                  for line in f.readlines()[1:]]
        with open(data_filename, encoding='utf-8') as f:
            self.negative_data_SpeechPeopleDisambiguationGrammarMatches = [line.strip().split('\t')[8]
                                  for line in f.readlines()[1:]]
        with open(data_filename, encoding='utf-8') as f:
            self.negative_data_ConversationalContext = [line.strip().split('\t')[9]
                                  for line in f.readlines()[1:]]
        '''

        with open(data_filename, encoding='utf-8') as f:
            self.negative_data = [line.rstrip('\n').split('\t')[2]
                                  for line in f.readlines()[1:]]
        with open(data_filename, encoding='utf-8') as f:
            self.negative_data_intent = [line.rstrip('\n').split('\t')[3]
                                  for line in f.readlines()[1:]]
                              
        with open(data_filename, encoding='utf-8') as f:
            self.negative_data_PreviousTurnDomain = [line.rstrip('\n').split('\t')[4]
                                  for line in f.readlines()[1:]]
        with open(data_filename, encoding='utf-8') as f:
            self.negative_data_TaskFrameStatus = [line.rstrip('\n').split('\t')[5]
                                  for line in f.readlines()[1:]]

        with open(data_filename, encoding='utf-8') as f:
            self.negative_data_TaskFrameEntityStates = [line.rstrip('\n').split('\t')[6]
                                  for line in f.readlines()[1:]]
        with open(data_filename, encoding='utf-8') as f:
            self.negative_data_TaskFrameGUID = [line.rstrip('\n').split('\t')[7]
                                  for line in f.readlines()[1:]]
        with open(data_filename, encoding='utf-8') as f:
            self.negative_data_SpeechPeopleDisambiguationGrammarMatches = [line.rstrip('\n').split('\t')[8]
                                  for line in f.readlines()[1:]]
        with open(data_filename, encoding='utf-8') as f:
            self.negative_data_ConversationalContext = [line.rstrip('\n').split('\t')[9]
                                  for line in f.readlines()[1:]]


        '''
        with open(data_filename, encoding='utf-8') as f:
            self.negative_data = [line.split('\t')[2]
                                  for line in f.readlines()[1:]]
        with open(data_filename, encoding='utf-8') as f:
            self.negative_data_intent = [line.split('\t')[3]
                                  for line in f.readlines()[1:]]
                              
        with open(data_filename, encoding='utf-8') as f:
            self.negative_data_PreviousTurnDomain = [line.split('\t')[4]
                                  for line in f.readlines()[1:]]
        with open(data_filename, encoding='utf-8') as f:
            self.negative_data_TaskFrameStatus = [line.split('\t')[5]
                                  for line in f.readlines()[1:]]

        with open(data_filename, encoding='utf-8') as f:
            self.negative_data_TaskFrameEntityStates = [line.split('\t')[6]
                                  for line in f.readlines()[1:]]
        with open(data_filename, encoding='utf-8') as f:
            self.negative_data_TaskFrameGUID = [line.split('\t')[7]
                                  for line in f.readlines()[1:]]
        with open(data_filename, encoding='utf-8') as f:
            self.negative_data_SpeechPeopleDisambiguationGrammarMatches = [line.split('\t')[8]
                                  for line in f.readlines()[1:]]
        with open(data_filename, encoding='utf-8') as f:
            self.negative_data_ConversationalContext = [line.split('\t')[9]
                                  for line in f.readlines()[1:]]
        '''

        #for debug 
        #print("negative data before preprocessing {}".format(self.negative_data))


        #extend extra negative data
        if extra_data_filename is not None:
            self._load_additional_negative_data(extra_data_filename)




        ## remove special character, extract ngram features from negative data    
        self._preprocess_negative_data()

        ## do not filter based on negative data 
        # self._filter_negative_data()

    def _preprocess_data(self):

        # convert to lower case
        self.positive_data = [q.lower() for q in self.positive_data]
        self.negative_data = [q.lower() for q in self.negative_data]


    def write_data_with_extra_columns(self, data_filename, keylist_filename):

        #? here positive_data should be renamed to data in the future
        num_pos_data = len(self.positive_data)
        print("samples: {}".format(num_pos_data))

        print("Writing postive keylist to file: {}".format(keylist_filename))
        with open(keylist_filename, 'w', encoding='utf-8') as f:
            for kw in set(self.keylist):
                f.write("{}\n".format(kw))

        
        # positive data format
        # query \t domain \t annotated_query


        # domain
        '''
        print("Writing all data to file: {}".format(data_filename))
        with open(data_filename, 'w', encoding='utf-8') as f:
            f.write("TurnNumber\tPreviousTurnDomain\tquery\tdomain\n")
            for query in tqdm(self.positive_data):
                f.write("0\t\t{}\tfiles\n".format(query))

            for query in tqdm(self.negative_data):
                f.write("0\t\t{}\tnot_files\n".format(query))

        '''
        #bellevue evaluation format
        # format(query, queries_domain[query], queries_annotated_queries[query], queries_intent[query]))
        print("Writing all data to file: {}".format(data_filename))
        id = 0
        with open(data_filename, 'w', encoding='utf-8') as fout:
            fout.write('\t'.join(['MessageText', 'JudgedDomain', 'JudgedConstraints', 'JudgedIntent']) + '\n')
            for item in tqdm(self.positive_data):


                #hanlde specila character
                item = item.replace('&', '&amp;')

                # for debug
                #print("{}".format(item))
                fout.write(item+'\n')



    def write_data(self, data_filename, neg_filename, pos_filename, keylist_filename, suffixkeylist_filename,  teams_positve_data_filename):
        num_pos_data = len(self.positive_data)
        num_neg_data = len(self.negative_data)
        print("Positive samples: {} | Negative samples {}".format(
            num_pos_data, num_neg_data))

        # num_dupl_queries = num_pos_data - len(set(self.positive_data))
        # print("Duplicate positive queries: {} :: {}%".format(num_dupl_queries, num_dupl_queries*100/num_pos_data))

        # with open("Duplicate_queries.txt", 'w', encoding='utf-8') as f:
        #     dup_queries = [query for query, cnt in Counter(self.positive_data).items() if cnt > 100]  
        #     for query in dup_queries:
        #         f.write('{}\n'.format(query)) 
        # 
        # 
        # 
        # 
         
        with open(teams_positve_data_filename, encoding='utf-8') as f:
            teams_pos_data = [line.rstrip('\n')
                                  for line in f.readlines()[1:]]



        print("Random tags selected from additional pool: {}, from random generated pool {}".format(
            self.additional_filetag_selection_cnt, self.random_filetag_selection_cnt))
        print("Retry attempts while selecting file names - filenames: {}".format(self._filename_gen_retry_list[:100]))
        if not TEST:
            print("Writing negative data to file: {}".format(neg_filename))
            with open(neg_filename, 'w', encoding='utf-8') as f:                    
                for query in tqdm(self.negative_data):
                    f.write("{}\n".format(query))

            print("Writing postive data to file: {}".format(pos_filename))
            with open(pos_filename, 'w', encoding='utf-8') as f:
                for query in tqdm(self.positive_data):
                    f.write("{}\n".format(query))

        print("Writing postive keylist to file: {}".format(keylist_filename))
        with open(keylist_filename, 'w', encoding='utf-8') as f:
            for kw in set(self.keylist):
                f.write("{}\n".format(kw))

        print("Writing postive keylist to file: {}".format(suffixkeylist_filename))
        with open(suffixkeylist_filename, 'w', encoding='utf-8') as f:
            for kw in set(self.filekeylistwithsuffix):
                f.write("{}\n".format(kw))

        print("Writing all data {} to file: {}".format(len(self.positive_data) +len(teams_pos_data) + len(self.negative_data), data_filename))
        with open(data_filename, 'w', encoding='utf-8') as f:
            # new header
            header = "id\tquery\tintent\tdomain\tQueryXml\n"


            f.write(header)
            #f.write("TurnNumber\tPreviousTurnIntent\tquery\tintent\tPreviousTurnDomain\tTaskFrameStatus\tTaskFrameEntityStates
#TaskFrameGUID\tSpeechPeopleDisambiguationGrammarMatches\tConversationalContext\n")


            print("generate positve data {}".format(len(self.positive_data)))

            for query in tqdm(self.positive_data):

                # positive files has extra doamin, intent/ annotation
                # skip them for domain traininf
                #f.write("0\t\t{}\tfiles\n".format(query))

                # ? here assume all patterns should be file_search
                # ? might need to filter more in the future
                f.write("0\t{}\tfile_search\tfiles\t{}\n".format((query.split('\t'))[0],(query.split('\t'))[2]))

            print("teams positve data {}".format(len(teams_pos_data)))
            for line in tqdm(teams_pos_data):

                # positive files has extra doamin, intent/ annotation
                # skip them for domain traininf
                #f.write("0\t\t{}\tfiles\n".format(query))

                # ? here assume all patterns should be file_search
                # ? might need to filter more in the future
                f.write("{}\n".format(line))


            # for debug
            #print("negative data {}".format(len(self.negative_data)))
            #print("{}".format(len(self.negative_data_intent)))
            #print("{}".format(len(self.negative_data_PreviousTurnDomain)))
            #print("{}".format(len(self.negative_data_TaskFrameStatus)))
            #print("{}".format(len(self.negative_data_TaskFrameEntityStates)))
            #print("{}".format(len(self.negative_data_TaskFrameGUID)))
            #print("{}".format(len(self.negative_data_SpeechPeopleDisambiguationGrammarMatches)))
            #print("{}".format(len(self.negative_data_ConversationalContext)))


            # negative data
            # extend to extra columns
            # for slot it does not have negative data so ignore it at first
            ##for query, intent, previousturnDomain, taskFrameStatus, taskFrameEntityStates, taskFrameGUID, speechPeopleDisambiguationGrammarMatches, ConversationalContext \
            ##         in tqdm(zip(self.negative_data,
            ##        self.negative_data_intent,
            ##        self.negative_data_PreviousTurnDomain,
            ##        self.negative_data_TaskFrameStatus,
            ##        self.negative_data_TaskFrameEntityStates,
            ##        self.negative_data_TaskFrameGUID,
            ##        self.negative_data_SpeechPeopleDisambiguationGrammarMatches,
            ##        self.negative_data_ConversationalContext
            ##        )):
            ##        f.write("0\t\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
            ##            query, intent, previousturnDomain, taskFrameStatus, taskFrameEntityStates, 
            ##            taskFrameGUID, speechPeopleDisambiguationGrammarMatches, ConversationalContext))

            #for query in tqdm(self.negative_data):
            #    f.write("0\t\t{}\tnot_files\n".format(query))


## data class
data = Data()

## xml value for each xml tag
## change to my own solution 
#all_tags_files = glob('./placeholder_tags/*.txt')
all_tags_files = glob('./placeholder_tags_chiecha/*.txt')
for tag_file in tqdm(all_tags_files):
    data.load_tags_data(tag_file)


#overtrigger tags
all_overtrigger_tags_files = glob('./overtrigger_tags/*.txt')
for tag_file in tqdm(all_overtrigger_tags_files):
    data.load_overtrigger_tags_data(tag_file)

## negative_corpus.txt
# for slot it does not have negative data so ignore it at first
##data.load_negative_data('negative_corpus.txt', 'additional_neg_data_in_query_form_chiecha.txt')


# using my own negative data then change to this one
#data.load_negative_data('mediacontrol_domain_train_after_filter_dedup.txt')


# positive patterns, comment it at first
# in the future, you can choose to use existing slot data combine together for contexual lu or generate data
# by positive patterns
## patterns.txt generated by Extract_patterns.py
## change to my own source
#data.load_positive_patterns('patterns.txt')
#data.load_positive_patterns('patterns_chiecha.txt')

##all_patterns_files = glob('./patterns/*.txt')
##data.load_patterns(all_patterns_files)


#kanshan extra pattern but slot reformated with _ by myself
##data.load_addtional_patterns('additional_patterns.txt')
##data.load_addtional_patterns('additional_patterns_1.txt')

#files search related patterns
# add variety for contact_name and to_contact_name
##data.load_addtional_patterns('patterns_FILES.txt')
##data.load_addtional_patterns('patterns_FILES_contact_name.txt')



# without keyword mapped to file_keyword
#data.append_attachment_patterns('patterns_EMAILSEARCH_attachment_after_bug_fix.txt','FILES', 'file_search')
# with keyword mapped to file_keyword
#data.append_attachment_patterns('patterns_EMAILSEARCH_attachment_add_keyword_after_bug_fix.txt','FILES', 'file_search')
#data.append_attachment_patterns('patterns_EMAILSEARCH_add_three_contact_name_slot_after_bug_fix.txt','FILES', 'file_search')
data.append_attachment_patterns('patterns_EMAILSEARCH_add_three_contact_name_orderref_after_bug_fix.txt','FILES', 'file_search')


# my extra patterns
#move it to pattern directory
#data.load_addtional_patterns('additional_patterns_chiecha.txt')


# for slot, no read all_patterns_files = glob('./patterns/*.txt') before really working on contexual lu
# so select max freq  = 5 then call it (number 5 does not matter)
#data.split_positive_patterns()
data.max_freq=5
data.split_patterns()


#data.get_positive_data()
data.get_data()









## output to my own location
#data.write_data(data_filename='Domain_Train.tsv', pos_filename='pos_data.tsv',
#                neg_filename='neg_data.tsv', keylist_filename='filekeys.txt')

#data.write_data(data_filename='files_domain_training_contexual_answer.tsv', pos_filename='pos_data_chiecha.tsv',
#                neg_filename='neg_data_chiecha.tsv', keylist_filename='filekeys_chiecha.txt')


#data.write_data(data_filename='files_intent_training_contexual_answer.tsv', pos_filename='pos_data_chiecha.tsv',
#                neg_filename='neg_data_chiecha.tsv', keylist_filename='filekeys_intent_chiecha.txt', 
#                suffixkeylist_filename = 'filekeys_suffix_intent_chiecha.txt'
#                )

#Positive_slot_corpus.txt
# if only wants to generate data for attachment pattern, leave it empty
# otherwsise, it should be files_slot_training.tsv
data.write_data(data_filename='files_slot_raining_contexual_answer.tsv', pos_filename='pos_data_chiecha.tsv',
                neg_filename='neg_data_chiecha.tsv', keylist_filename='filekeys_slot_chiecha.txt', 
                suffixkeylist_filename = 'filekeys_suffix_slot_chiecha.txt',
                teams_positve_data_filename='Positive_slot_corpus.txt'
                )
                



#output not only domain trainngi data
##data.write_data_with_extra_columns(data_filename='files_domain_training_contexual_answer.tsv', keylist_filename='filekeys_chiecha.txt')

# writing data to pos_data file

###############################
# Test methods
###############################

test = Data()
# for tag_file in all_tags_files:
#     test.load_tags_data(tag_file)

# test.load_positive_patterns('patterns.txt')

# queries = test.replace_tag_randomly('show me <filename> word <contactname> was working on', 7)
# pprint(queries)
# pprint(test.keylist)

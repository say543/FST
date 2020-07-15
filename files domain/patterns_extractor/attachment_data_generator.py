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


class Data(object):

    def __init__(self):
        # positive data for data tag augmentation
        self.posdata_scale_multiplier = 1000  # multiplier to amplify the positive_dataset
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

        self.tags = {}
        self.patterns = {}  # pattern: frequency
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

        ## 'filekeyword', 'fileskeyword', 'filename' being used when generating positive data
        ## source can from *.txt or nltk
        self.keylist = []  # list to accumulate file keywords

        self.negative_data = []
        self.negative_ngrams = defaultdict(int)  # dictionary for negative data lookup for filetags
        self._filename_gen_retry_list = []

        ## nltk library
        ## https://www.nltk.org/book/ch02.html
        # https://medium.com/pyladies-taiwan/nltk-%E5%88%9D%E5%AD%B8%E6%8C%87%E5%8D%97-%E4%BA%8C-%E7%94%B1%E5%A4%96%E8%80%8C%E5%85%A7-%E5%BE%9E%E8%AA%9E%E6%96%99%E5%BA%AB%E5%88%B0%E5%AD%97%E8%A9%9E%E6%8B%86%E8%A7%A3-%E4%B8%8A%E6%89%8B%E7%AF%87-e9c632d2b16a
        self.vocab = nltk.corpus.words.words()

        self.da = DataAugmentation()


        # read filetype
        # using uwp one to remove picture file
        self.filetype = []
        self.fileboost = []
        #with codecs.open('..\\resource\\lexicons\\file_type_domain_boost.txt', 'r', 'utf-8') as fin:
        with codecs.open('..\\resource\\lexicons\\file_type_domain_boost_UWP.txt', 'r', 'utf-8') as fin:
            for line in fin:
                line = line.strip()
                if line.lower() == 'documents' or line.lower() == 'document' or line.lower() == 'file' or line.lower() == 'files':
                    self.fileboost.append(line)
                else:
                    self.filetype.append(line)

        # comment this since prefer original slot for evaluation
        self.domain_slot_process = defaultdict(set)
        self.domain_slot_process['EMAILSEARCH'].add('message_type')
        self.domain_slot_process['EMAILSEARCH'].add('attachment_type')

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



        # tag mapped logic
        self.tag_originalTag = defaultdict(defaultdict)
        self.tag_originalTag['CALENDAR']['title'] = 'title'
        self.tag_originalTag['PEOPLE']['peopleattribute'] = 'people_attribute'
        self.tag_originalTag['TEAMSMESSAGE']['keyword'] = 'keyword'
        self.tag_originalTag['EMAIL']['emailsubject'] = 'email_subject'
        self.tag_originalTag['EMAIL']['message'] = 'message'
        self.tag_originalTag['EMAIL']['keyword'] = 'keyword'
        self.tag_originalTag['NOTE']['notetext'] = 'note_text'
        self.tag_originalTag['REMINDER']['remindertext'] = 'reminder_text'
        self.tag_originalTag['FILES']['filekeyword'] = 'file_keyword'
        self.tag_originalTag['FILES']['filename'] = 'file_name'
        '''


    def load_tags_data(self, file):
        tag = os.path.basename(file).replace('.txt', '')

        ## remain head / tailing spaces
        with open(file, encoding='utf-8') as f:
            values = [val.strip() for val in f.readlines()]

        if tag not in self.tags:
            self.tags[tag] = Tag(tag, values)
        else:
            raise Exception('Duplicate tags found for {}'.format(tag))

    def _augment_patterns(self):
        similar_patterns_dict = {}
        for pattern, freq in self.patterns.items():
            similar_patterns = self.da.get_similar_patterns(pattern)
            similar_patterns_dict.update({sim_pattern: freq for sim_pattern in similar_patterns})

        self.patterns.update(similar_patterns_dict)

    def _load_additional_positive_patterns(self, filename, default_freq):
        with open(filename, encoding='utf-8') as f:
            additional_patterns = {pattern.strip(): default_freq for pattern in f.readlines()}
            self.patterns.update(additional_patterns)
            self.additional_patterns_loaded = True



    def load_patterns(self, all_patterns_files, **kwargs):


        for file in tqdm(all_patterns_files):

            print("Loading patterns from {}".format(file))


            with open(file, encoding='utf-8') as f:
                pattern_data = [line.strip() for line in f.readlines()]




                # might ne empty
                if not pattern_data:
                    raise Exception('No positive patterns loaded!')

                # freq sorted by high to lower, so using the first one to record frequency
                self.max_freq = int(pattern_data[0].split('\t')[1])


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

                    self.patterns[(p.split('\t')[0])] = int(p.split('\t')[1])

                    self.patterns_domain[(p.split('\t')[0])] = domain

                    self.patterns_annotated_queries[(p.split('\t')[0])] = p.split('\t')[3]
                    self.patterns_intent[(p.split('\t')[0])] = p.split('\t')[4]

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
        print("Loading additional patterns.")
        #self._load_additional_positive_patterns('additional_patterns.txt', self.max_freq)
        self._load_additional_positive_patterns(file, self.max_freq)
        # self._augment_patterns()     
   
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

    def _get_random_generated_filetag(self):
        # This fn checks the effects of filename/keyword selection
        random_filename = []
        num_words = random.randint(1, 4)

        if not self.negative_ngrams:
            raise Exception("Negative patterns not loaded")

        attempt = 0
        max_attempts = 10
        while attempt < max_attempts:

            ## generate random words from self.vocad(nltk corpus)
            for _ in range(num_words):
                random_filename.extend(random.choices(self.vocab, k=1))

            generated_filename = " ".join(random_filename)

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
        '''
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
        '''



        # finding tags in a query
        tags = re.findall(r'<(.*?)>', pattern)

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


                if tag not in self.domain_slot_process[domain]:
                    #raise Exception(
                    #    'Unknown tag {} found in the query pattern {}'.format(tag, pattern))

                    print('Unknown tag {} found in the query pattern {}, ignore it'.format(tag, pattern))
#                if tag not in self.tags:
#                    raise Exception(
#                        'Unknown tag {} found in the query pattern {}'.format(tag, pattern))



                else:

                    '''
                    # file keyword routine, do not do it for attachment right now
                    # share <filekeyword> -> share OKR
                    # do not use probably to replace
                    # use keyphrases to replace

                    ## _filetag_selection can choose from Additional Tag , Randpm Tag, Original tag
                    #filetag_src = self._filetag_selection()
                    random_tag = self.tags['additionalfilenames'].get_random_value()
                    self._add_tagvalue_to_keylist(random_tag)
                    '''

                    
                    if (tag == 'message_type'):
                        inRangeIndex = random.randint(0, len(self.fileboost)-1)

                        random_tag = self.fileboost[inRangeIndex]
                        query = query.replace("<{}>".format(tag), random_tag)
                        # file boost xml is removed here
                        annotation = annotation.replace("<{}>".format(tag), 
                            " {} ".format(random_tag))

                    elif (tag == 'attachment_type'):
                        inRangeIndex = random.randint(0, len(self.filetype)-1)

                        random_tag = self.filetype[inRangeIndex]
                        query = query.replace("<{}>".format(tag), random_tag)
                        annotation = annotation.replace("<{}>".format(tag), 
                            "<{}> {} </{}>".format(tag, random_tag, tag))
                    
                    '''
                    query = query.replace("<{}>".format(tag), random_tag)
                    #annotation = annotation.replace("<{}>".format(tag), "<{}>".format(self.tag_originalTag[domain][tag]))

                    annotation = annotation.replace("<{}>".format(tag), 
                        "<{}> {} </{}>".format(tag, random_tag, tag))
                    '''
            ##for debug
            print("query: {}".format(query))


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
                # pattern, freq*freq_multiplier)) # Removed dataset freq to give equal importance
                pattern, freq_multiplier)

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

    def get_data(self, high_freq_multiplier=8, low_freq_multiplier=1):
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
        self.negative_data = list(set([self._clean_query(query) for query in tqdm(self.negative_data)]))

    def _filter_negative_data(self):
        self.negative_data = self._get_random_negative_samples(int(len(self.negative_data) * 0.7))
        print("Filtering negative data to remove overlap with file keyword list.")
        original_corpus_len = len(self.negative_data)
        random_filekeys_sample = random.choices(list(self.keylist), k=70000)

        for filekey in tqdm(random_filekeys_sample):
            self.negative_data = [query for query in self.negative_data if filekey not in query]

        print("{} queries were filtered".format(original_corpus_len - len(self.negative_data)))

    def load_negative_data(self, data_filename):

        with open(data_filename, encoding='utf-8') as f:
            self.negative_data = [line.strip().split('\t')[2]
                                  for line in f.readlines()[1:]]

        ## remove special character, extract ngram features from negative data    
        self._preprocess_negative_data()

        ## do not filter based on negative data 
        # self._filter_negative_data()

    def _preprocess_data(self):

        # convert to lower case
        self.positive_data = [q.lower() for q in self.positive_data]
        self.negative_data = [q.lower() for q in self.negative_data]


    def write_data(self, data_filename, keylist_filename):

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



    '''
    def write_data(self, data_filename, neg_filename, pos_filename, keylist_filename):
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

        print("Writing all data to file: {}".format(data_filename))
        with open(data_filename, 'w', encoding='utf-8') as f:
            f.write("TurnNumber\tPreviousTurnDomain\tquery\tdomain\n")
            for query in tqdm(self.positive_data):
                f.write("0\t\t{}\tfiles\n".format(query))

            for query in tqdm(self.negative_data):
                f.write("0\t\t{}\tnot_files\n".format(query))
    '''

## data class
data = Data()

## xml value for each xml tag
## change to my own solution 
#all_tags_files = glob('./placeholder_tags/*.txt')
all_tags_files = glob('./placeholder_tags_chiecha/*.txt')

for tag_file in tqdm(all_tags_files):
    data.load_tags_data(tag_file)

## negative_corpus.txt
## coming from my negative examples
## but have not been preprocessing and deduplicting
#data.load_negative_data('negative_corpus.txt')

## patterns.txt generated by Extract_patterns.py
## change to my own source
#data.load_positive_patterns('patterns.txt')
#data.load_positive_patterns('patterns_chiecha.txt')

all_patterns_files = glob('./patterns_attachment/*.txt')
data.load_patterns(all_patterns_files)

#commend additional pattern at first
#data.load_addtional_patterns('additional_patterns_chiecha.txt')

## frequency is not being used right know so this function does nothing
#data.split_positive_patterns()
data.split_patterns()


#data.get_positive_data()
data.get_data()


data.write_data(data_filename='AttachmentData.tsv', keylist_filename='filekeys_chiecha.txt')



'''
## output to my own location
#data.write_data(data_filename='Domain_Train.tsv', pos_filename='pos_data.tsv',
#                neg_filename='neg_data.tsv', keylist_filename='filekeys.txt')

data.write_data(data_filename='Domain_Train_chiecha.tsv', pos_filename='pos_data_chiecha.tsv',
                neg_filename='neg_data_chiecha.tsv', keylist_filename='filekeys_chiecha.txt')


data.write_data(data_filename='Domain_Train_chiecha.tsv', pos_filename='pos_data_chiecha.tsv',
                neg_filename='neg_data_chiecha.tsv', keylist_filename='filekeys_chiecha.txt')
'''
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

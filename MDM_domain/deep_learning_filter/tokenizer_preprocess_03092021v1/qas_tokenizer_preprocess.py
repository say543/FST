from glob import glob
import pandas as pd
import re
from pprint import pprint
import string
from collections import defaultdict
import json
import os
import csv
from shutil import copyfile
import codecs




# clean query and get pattern
def clean_query(query):

    ## string punctuation
    ## https://www.geeksforgeeks.org/string-punctuation-in-python/
    # excluding <>, _ since _ will insdie filt tag  and / will be insdie tag
    
    punctuation = [p for p in string.punctuation if p != '<' and p != '>' and p != '_' and p != '/']

    ## remove exclude words
    ## ? might be a can be removed as well
    exclude_words = ['hey', 'cortana', 'the']

    ## remove punctutation
    query = ''.join(ch for ch in query if ch not in set(punctuation))
    query_filtered = [w for w in word_tokenize(query.lower()) if w not in exclude_words]

    ## "< " , " >" remove extra space 
    return " ".join(query_filtered).replace("< ","<").replace(" >",">")

'''
def clean_query(query):

    ## string punctuation
    ## https://www.geeksforgeeks.org/string-punctuation-in-python/
    # excluding <>
    # so _ will be replace  eg: file_keyword becomes filekeyword
    punctuation = [p for p in string.punctuation if p != '<' and p != '>']

    ## remove exclude words
    ## ? might be a can be removed as well
    exclude_words = ['hey', 'cortana', 'the']

    ## remove punctutation
    query = ''.join(ch for ch in query if ch not in set(punctuation))
    query_filtered = [w for w in word_tokenize(query.lower()) if w not in exclude_words]

    ## "< " , " >" remove extra space 
    return " ".join(query_filtered).replace("< ","<").replace(" >",">")
'''

def process_tagged_queries(queries, annotated_queries, intents, domain, DomainToSlotsProcess):
    # assert(len(annotated_queries)==len(queries), "Invalid query, annotation set")
    
    # if needed, change this to list to also store list of queries under a pattern

    ## deduplicate patterns
    pattern_queries = defaultdict(int)
    pattern_queries_to_annotated_queries = defaultdict()
    pattern_queries_to_intent = defaultdict()

    
    tags = ['sharetarget_name', 'file_name', 'file_type', 'sharetarget_type', 'order_ref', 'to_contact_name', 'date', 'files_keyword',
            'file_action', 'file_action_context', 'file_keyword', 'file_folder', 'meeting_starttime', 'contact_name', 'file_recency']


    ## dictionary of list
    ## key : xml_name
    ## value : list of possible values inside <> </>
    tag_values = defaultdict(list)


    for query, ann_query, intent in zip(queries, annotated_queries, intents):
        # print(query)

        ## using originla query to create new patterns
        new_query = query
        new_annotation = ann_query

        # for debug
        #print('annotated query {}'.format(new_annotation)) 

        ## extract all constraints (XML pair) from ann_query
        # new routine ,extract 
        xmlpairs = re.findall("(<.*?>.*?<\/.*?>)", ann_query)

        for xmlpair in xmlpairs:
            # extra type and value for xml tag
            xmlTypeEndInd = xmlpair.find(">")

            xmlType = xmlpair[1:xmlTypeEndInd]

            xmlValue = xmlpair.replace("<"+xmlType+">", "")
            xmlValue = xmlValue.replace("</"+xmlType+">", "")
            xmlValue = xmlValue.strip()

            # only extra certain slots to form pattens
            if domain in DomainToSlotsProcess and xmlType.lower() in DomainToSlotsProcess[domain]:

                
                #if xmlType.lower() == 'message_type':
                #    print('{}'.format(xmlValue))
                #    print('{}'.format(new_query))

                tag_values[xmlType].extend(xmlValue)

                #######################
                # version 1
                #######################
                '''
                if xmlType == 'message_type' or xmlType == 'keyword':
                    new_query = new_query.replace(xmlValue, '')
                    new_annotation = new_annotation.replace(xmlpair, '')
                else:
                    new_query = new_query.replace(xmlValue, '<{}>'.format(xmlType.lower()))

                    #if xmlType.lower() == 'message_type':
                    #    print('{}'.format(new_query))
                    #    print('{}'.format(clean_query(new_query)))

                    new_annotation = new_annotation.replace(xmlpair, '<{}>'.format(xmlType.lower()))
                '''

                #######################
                # version 2
                #######################
                new_query = new_query.replace(xmlValue, '<{}>'.format(xmlType.lower()))

                # fro debug
                #if xmlType.lower() == 'message_type':
                #    print('{}'.format(new_query))
                #    print('{}'.format(clean_query(new_query)))

                new_annotation = new_annotation.replace(xmlpair, '<{}>'.format(xmlType.lower()))

        # old routine, all tags being processed
        '''
        for tag in tags:
            keywords = re.findall(rf'<{tag}>(.+?)<\/{tag}>', ann_query)
            
            #collecting the variables to fill tag
            ## ? keyword do not preprocessing  eg:  3S api spec do not do tokenization
            tag_values[tag].extend([kw.lower() for kw in keywords])
            
            # stripping query of tags
            ## using originla query to create new annotated query
            ## eg: search for 3s api deck, => search for <file_keyword> deck since 3s api is tagged as slot in annotated_query   
            for kw in keywords:
                new_query = query.replace(kw, '<{}>'.format(tag))

        '''

        #for debug
        #print('{}'.format(new_query))
        #print('{}'.format(clean_query(new_query)))

        pattern_queries[clean_query(new_query)] += 1

        pattern_queries_to_annotated_queries[clean_query(new_query)] = clean_query(new_annotation)
        pattern_queries_to_intent[clean_query(new_query)] = intent

    print('-I-: Given queries {}, domain {}, total patterns are {}'.format(len(queries),domain, len(pattern_queries)))
    return pattern_queries, tag_values,pattern_queries_to_annotated_queries, pattern_queries_to_intent

## read *.tsv name and output .txt fils
## txt format: each filename / keypharse will be single line
## tag means slot value here. rename it since it is confusing
'''
def extract_additional_tags(filename):
    tags = []
    df = pd.read_csv(filename, sep='\t', encoding="utf-8")
    for val in df.List.values:
        tags.extend(val.split(';'))
        
    with open(filename.replace(".tsv", ".txt"), 'w', encoding='utf-8') as f:
        for tag in tags:
            f.write('{}\n'.format(tag))
        
# step1
extract_additional_tags("additionalfilenames.tsv")
extract_additional_tags("additionalfilenameskeyphrases.tsv")
'''


# step2

# customized logic processing
DomainToSlotsProcess = defaultdict(set)
DomainToSlotsProcess['EMAILSEARCH'].add('message_type')
DomainToSlotsProcess['EMAILSEARCH'].add('attachment_type')
# attachment will be confused with attachment_type when replacement. leave it to data generation
#DomainToSlotsProcess['EMAILSEARCH'].add('attachment')
# add keyword for contexul lu patterns extraction
DomainToSlotsProcess['EMAILSEARCH'].add('keyword')




'''
DomainToSlotsProcess['CALENDAR'].add('title')
#DomainToSlotsProcess['PEOPLE'].add('peopleattribute')
DomainToSlotsProcess['PEOPLE'].add('people_attribute')
DomainToSlotsProcess['TEAMSMESSAGE'].add('keyword')
#DomainToSlotsProcess['EMAIL'].add('emailsubject')
DomainToSlotsProcess['EMAIL'].add('email_subject')
DomainToSlotsProcess['EMAIL'].add('message')
DomainToSlotsProcess['EMAIL'].add('keyword')
#DomainToSlotsProcess['NOTE'].add('notetext')
DomainToSlotsProcess['NOTE'].add('note_text')
#DomainToSlotsProcess['REMINDER'].add('remindertext')
DomainToSlotsProcess['REMINDER'].add('reminder_text')
DomainToSlotsProcess['FILES'].add('file_keyword')
DomainToSlotsProcess['FILES'].add('file_name')
'''




# change to another place
data_files = glob('./original_parse_data/*.tsv')

## deuplication pattern
#all_patterns = defaultdict(int)
## deduplication tag (slot value)
#all_tags = defaultdict(list)



#target = pd.read_csv('target.tsv', sep='\t', encoding="utf-8")
#target = pd.read_csv('qd_uber_joint_dnn_featurizer_nonjson.uber.FeatureIds.output.txt', sep='\t', encoding="utf-8", dtype={
#    'MessageId': object, 'Frequency': object, 'ConversationContext': object, 'SelectionIgnore': object})



#for index, row in target.iterrows():
#    print(target.head())

tokensIdForQueries = []
tokensForQueries = []
labelIdsForQueries = []
#with codecs.open('qd_uber_joint_dnn_featurizer_nonjson.uber.FeatureIds.output.txt', 'r', 'utf-8') as fin:
with codecs.open('E:/CoreScienceDataStaging/TestResults/Evaluation/qd_uber_joint_dnn_featurizer_nonjson.uber.FeatureIds.output.txt', 'r', 'utf-8') as fin:    
    for line in fin:
        line = line.strip();
        if not line:
            continue;
        linestrs = line.split("\t");

        i = 0
        while i < len(linestrs):
            ele = linestrs[i]



            if ele.lower() == 'tokens':


                # for deubg
                #print("{}".format(ele.lower()))

                tokensForQuery = []


                # skip two 
                num_tokens = linestrs[i+2]
                i = i+3

                # for deubg
                print("num_tokens\t{}".format(num_tokens))

                for j in range(0, int(num_tokens)):


                    token = linestrs[i+j*4]

                    # for deubg 
                    #print("token\t{}".format(token))

                    tokensForQuery.append(token)
        
                i = i+ int(num_tokens) *4       
                tokensForQueries.append(tokensForQuery)
            elif ele.lower() == 'input_ids':


                # for deubg
                #print("{}".format(ele.lower()))

                tokensIdForQuery = []


                # skip two 
                num_tokens = linestrs[i+1]
                i = i+3

                # for deubg
                print("num_tokensid\t{}".format(num_tokens))

                for j in range(0, int(num_tokens)):


                    tokenid = linestrs[i+j*4]

                    # for deubg 
                    #print("tokenid\t{}".format(tokenid))

                    tokensIdForQuery.append(tokenid)
        
                i = i+ int(num_tokens) *4       
                tokensIdForQueries.append(tokensIdForQuery)

            #elif ele.lower() == 'slot_dnn_tag':


            #    # for deubg
            #    #print("{}".format(ele.lower()))

            #    labelIdsForQuery = []


            #    # skip two 
            #    num_tokens = linestrs[i+1]
            #    i = i+3

            #    # for deubg
            #    print("num_labelid\t{}".format(num_tokens))

            #    for j in range(0, int(num_tokens)):


            #        labelid = linestrs[i+j*4]

            #        # for deubg 
            #        #print("tokenid\t{}".format(tokenid))

            #        labelIdsForQuery.append(labelid)
        
            #    i = i+ int(num_tokens) *4       
            #    labelIdsForQueries.append(labelIdsForQuery)

            elif ele.lower() == 'slot_output':


                # for deubg
                #print("{}".format(ele.lower()))

                labelIdsForQuery = []


                # skip three 
                num_tokens = linestrs[i+1]
                i = i+4

                # for deubg
                print("num_labelid\t{}".format(num_tokens))

                for j in range(0, int(num_tokens)):


                    labelid = linestrs[i+j*4]

                    # for deubg 
                    #print("tokenid\t{}".format(tokenid))

                    labelIdsForQuery.append(labelid)
        
                i = i+ int(num_tokens) *4       
                labelIdsForQueries.append(labelIdsForQuery)
            else:
                i = i+1

#output_file = 'FeatureIds.output.txt'
output_file = 'E:/mdm_data_analysis/deep_learning_filter/tokenizer_preporcess_03092021v1/FeatureIds.output.txt'
with codecs.open(output_file, 'w', 'utf-8') as fout:

    # tokensForquery/ tokensIdForQuery , including SEP and CLS
    # labelIdsForQuery, excluding SEP and CLS

    for i, (tokensForQuery, tokensIdForQuery, labelIdsForQuery) in enumerate(zip(tokensForQueries, tokensIdForQueries,labelIdsForQueries)):
        
        #print(type(list(tokensIdForQuery)))
        tokensIdForQuery = list(tokensIdForQuery)
        #print(tokensIdForQuery[0])

        tokensForQuery = list(tokensForQuery)
        labelIdsForQuery = list(labelIdsForQuery)
        fout.write(" ".join(tokensForQuery) + '\t'+ " ".join(tokensIdForQuery)+ '\t' +" ".join(labelIdsForQuery)+'\r\n');

        #fout.write(" ".join(tokensForQuery[0]) + '\t'+ " ".join(tokensIdForQuery[0])+'\r\n');


#print(target.head())
#for col in target.columns: 
#    print(col) 



# column type
#print(target.info())


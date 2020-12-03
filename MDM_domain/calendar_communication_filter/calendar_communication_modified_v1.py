from glob import glob
import pandas as pd
import re
from pprint import pprint
import string
from nltk.tokenize import word_tokenize
from collections import defaultdict
import json
import os
import csv



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
target = pd.read_csv('target.tsv', sep='\t', encoding="utf-8", dtype={
    'MessageId': object, 'Frequency': object, 'ConversationContext': object, 'SelectionIgnore': object})


# column type
#print(target.info())

conversaionidlist = target['ConversationId'].values.tolist()
conversaionidSet=set(conversaionidlist)

#print(target['DatasetFile'].values)

ConversationContextlist = target['ConversationContext'].values.tolist()
ConversationContextSet=set(ConversationContextlist)

#print(target['ConversationContext'].values)

dataSetfilelist = target['DatasetFile'].values.tolist()
dataSetfileSet=set(dataSetfilelist)





#print('update dataset list {}'.format(dataSetfileSet))


for file in data_files[:]:
    ##     read one cvs file and filter files query
    #print('file name {}'.format(os.path.basename(file)))

    # check the dataset
    filebasename = os.path.basename(file)
    if filebasename in target['DatasetFile'].values:
        print('file name {}'.format(filebasename))

        #update_df = pd.read_csv(file, sep='\t', encoding="utf-8")
        update_df = pd.read_csv(file, sep='\t', encoding="utf-8", dtype={'SelectionIgnore': object})

        #update_df = original_df.copy()

        subset_target = target[target.DatasetFile == filebasename].copy()



        #https://stackoverflow.com/questions/24036911/how-to-update-values-in-a-specific-row-in-a-python-pandas-dataframe
        # https://stackoverflow.com/questions/49928463/python-pandas-update-a-dataframe-value-from-another-dataframe
        # v1
        # go through subset and update annotation
        #for index, row in subset_target.iterrows():
        '''
        subset_target.set_index('ConversationId', inplace=True)
        update_df.set_index('ConversationId', inplace=True)
        #subset_target.set_index('ConversationId')
        #update_df.set_index('ConversationId')
        update_df.update(subset_target)
        #update_df = update_df.reset_index().copy()  # to recover the initial structure

        #update_df.to_csv(filebasename+'_updated.tsv', header=None, index=None, sep='\t')
        update_df.to_csv(filebasename, sep='\t')
        '''


        #https://stackoverflow.com/questions/24036911/how-to-update-values-in-a-specific-row-in-a-python-pandas-dataframe
        # https://stackoverflow.com/questions/49928463/python-pandas-update-a-dataframe-value-from-another-dataframe
        # v2
        # go through subset and update annotation
        #for index, row in subset_target.iterrows():
        
        subset_target.set_index(['ConversationId', 'MessageId'], inplace=True)
        update_df.set_index(['ConversationId', 'MessageId'], inplace=True)
        #subset_target.set_index('ConversationId', inplace=True)
        #update_df.set_index('ConversationId', inplace=True)
        #subset_target.set_index('ConversationId')
        #update_df.set_index('ConversationId')
        update_df.update(subset_target)
        #update_df = update_df.reset_index().copy()  # to recover the initial structure

        #https://stackoverflow.com/questions/21147058/pandas-to-csv-output-quoting-issue
        # remove quote issues
        #update_df.to_csv(filebasename+'_updated.tsv', header=None, index=None, sep='\t')
        update_df.to_csv(filebasename, sep='\t', quoting=csv.QUOTE_NONE)        


        # for debug
        #if filebasename == 'Reporting_TVS_AllDomain_SR_20200401-20200430.tsv':
        #    print(update_df['ConversationContext'].values)


        # https://stackoverflow.com/questions/49928463/python-pandas-update-a-dataframe-value-from-another-dataframe
        '''
        update_df = pd.concat([update_df,subset_target]).drop_duplicates(['ConversationId'],keep='last')
        update_df.to_csv(filebasename+'_updated.tsv', header=None, index=None, sep='\t')
        '''

#print(df.head())
#for col in df.columns: 
#    print(col) 




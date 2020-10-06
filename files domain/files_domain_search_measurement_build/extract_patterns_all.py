from glob import glob
import pandas as pd
import re
from pprint import pprint
import string
from nltk.tokenize import word_tokenize
from collections import defaultdict
import json


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

        ## extract all constraints (XML pair) from ann_query
        # new routine ,extract 
        xmlpairs = re.findall("(<.*?>.*?<\/.*?>)", ann_query)
        
        # need to do to_contact_name first then contact_name
        # so reverse order
        #for xmlpair in xmlpairs:
        for xmlpair in sorted(xmlpairs, reverse=True):
            # extra type and value for xml tag
            xmlTypeEndInd = xmlpair.find(">")

            xmlType = xmlpair[1:xmlTypeEndInd]

            xmlValue = xmlpair.replace("<"+xmlType+">", "")
            xmlValue = xmlValue.replace("</"+xmlType+">", "")
            xmlValue = xmlValue.strip()

            # only extra certain slots to form pattens
            if domain in DomainToSlotsProcess and xmlType.lower() in DomainToSlotsProcess[domain]:
                tag_values[xmlType].extend(xmlValue)
                new_query = new_query.replace(xmlValue, '<{}>'.format(xmlType.lower()))
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
# add contact / to_contact_name for files only
# but to_contact_name needs to process earlier than contact_name to prevent problem
DomainToSlotsProcess['FILES'].add('contact_name')
DomainToSlotsProcess['FILES'].add('to_contact_name')






# change to another place
data_files = glob('./original_parse_data/*.tsv')

## deuplication pattern
#all_patterns = defaultdict(int)
## deduplication tag (slot value)
#all_tags = defaultdict(list)

## ? not sure what pos_queires_cnt means
#pos_queries_cnt = 0
dfs = []
for file in data_files[:]:
##     read one cvs file and filter files query
    temp = pd.read_csv(file, sep='\t', encoding="utf-8")
    dfs.append(temp)

df = pd.concat(dfs, axis=0, ignore_index=True)


#print(df.head())
#for col in df.columns: 
#    print(col) 


df.sort_values(by=['JudgedDomain'])
#print(df)




for domain in ['FILES', 'CALENDAR', 'PEOPLE', 'EMAIL', 'TEAMSMESSAGE', 'NOTE', 'REMINDER']:


    ## deuplication pattern
    all_patterns = defaultdict(int)
    ## deduplication tag (slot value)
    all_tags = defaultdict(list)

    ## ? not sure what pos_queires_cnt means
    pos_queries_cnt = 0

    target_df = df[df.JudgedDomain == domain]
    
##     ? not sure if data frame will perform deduplication
    pos_queries_cnt += len(target_df)
    
    patterns_dict, tag_values, pattern_queries_to_annotated_queries, pattern_queries_to_intent = process_tagged_queries(target_df.MessageText, target_df.JudgedConstraints, target_df.JudgedIntent, \
        domain, DomainToSlotsProcess)
    
    #collecting all the tags
    for tag, values in tag_values.items():
        all_tags[tag].extend(values)

    # collecting all patterns
##    also with patterns's frequency
    for pattern, count in patterns_dict.items():
        all_patterns[pattern] += count


    ## ouput patterns and frequency
    ## i output to a different locations to verify
    with open('./patterns/'+ 'patterns_'+ domain+'.txt', 'w', encoding='utf-8') as f:
        for pattern, count in reversed(sorted(all_patterns.items(), key=lambda x: x[1])):
            f.write('{}\t{}\t{}\t{}\t{}\n'.format(pattern, count, domain, pattern_queries_to_annotated_queries[pattern],\
                pattern_queries_to_intent[pattern]))

    ## if needing to output tags, open here
    ##print('Total postive queries found: {} | Total patterns: {}'.format(pos_queries_cnt, len(all_patterns)))
    ##for tag, values in all_tags.items():
    ##    with open('./placeholder_tags_chiecha/{}.txt'.format(tag.replace("_","")), 'w', encoding='utf-8') as f:
    ##        for value in set(values):
    ##            f.write(value+'\n')




# with open('patterns.txt', 'w', encoding='utf-8') as f:
#     for pattern, count in reversed(sorted(all_patterns.items(), key=lambda x: x[1])):
#         f.write('{}\t{}\n'.format(pattern, count))

# print('Total postive queries found: {} | Total patterns: {}'.format(pos_queries_cnt, len(all_patterns)))

# for tag, values in all_tags.items():
#     with open('./placeholder_tags/{}.txt'.format(tag.replace("_","")), 'w', encoding='utf-8') as f:
#         for value in set(values):
#             f.write(value+'\n')
        

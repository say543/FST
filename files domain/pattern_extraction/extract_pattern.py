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
    # excluding <>
    punctuation = [p for p in string.punctuation if p != '<' and p != '>']

    ## remove exclude words
    ## ? might be a can be removed as well
    exclude_words = ['hey', 'cortana', 'the']

    ## remove punctutation
    query = ''.join(ch for ch in query if ch not in set(punctuation))
    query_filtered = [w for w in word_tokenize(query.lower()) if w not in exclude_words]

    ## "< " , " >" remove extra space 
    return " ".join(query_filtered).replace("< ","<").replace(" >",">")


def process_tagged_queries(queries, annotated_queries):
    # assert(len(annotated_queries)==len(queries), "Invalid query, annotation set")
    
    # if needed, change this to list to also store list of queries under a pattern

    ## deduplicate patterns
    pattern_queries = defaultdict(int)
    
    tags = ['sharetarget_name', 'file_name', 'file_type', 'sharetarget_type', 'order_ref', 'to_contact_name', 'date', 'files_keyword',
            'file_action', 'file_action_context', 'file_keyword', 'file_folder', 'meeting_starttime', 'contact_name', 'file_recency']

    ## dictionary of list
    ## key : xml_name
    ## value : list of possible values inside <> </>
    tag_values = defaultdict(list)


    for query, ann_query in zip(queries, annotated_queries):
        # print(query)

        ## using originla query to create new patterns
        new_query = query

        ## extract all constraints (XML pair) from ann_query
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
        pattern_queries[clean_query(new_query)] += 1
    print('-I-: Given queries {}, total patterns are {}'.format(len(queries), len(pattern_queries)))
    return pattern_queries, tag_values

## read *.tsv name and output .txt fils
## txt format: each filename / keypharse will be single line
## tag means slot value here. rename it since it is confusing
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



# step2



data_files = glob('./csds_data/*.tsv')

## deuplication pattern
all_patterns = defaultdict(int)
## deduplication tag (slot value)
all_tags = defaultdict(list)

## ? not sure what pos_queires_cnt means
pos_queries_cnt = 0

for file in data_files[:]:
##     read one cvs file and filter files query
    df = pd.read_csv(file, sep='\t', encoding="utf-8")
    files_df = df[df.JudgedDomain == 'FILES']
    
##     ? not sure if data frame will perform deduplication
    pos_queries_cnt += len(files_df)
    
    patterns_dict, tag_values = process_tagged_queries(files_df.MessageText, files_df.JudgedConstraints)
    
    #collecting all the tags
    for tag, values in tag_values.items():
        all_tags[tag].extend(values)

    # collecting all patterns
##    also with patterns's frequency
    for pattern, count in patterns_dict.items():
        all_patterns[pattern] += count


## ouput patterns and frequency
## i output to a different locations to verify
with open('patterns_chiecha.txt', 'w', encoding='utf-8') as f:
    for pattern, count in reversed(sorted(all_patterns.items(), key=lambda x: x[1])):
        f.write('{}\t{}\n'.format(pattern, count))

print('Total postive queries found: {} | Total patterns: {}'.format(pos_queries_cnt, len(all_patterns)))

for tag, values in all_tags.items():
    with open('./placeholder_tags_chiecha/{}.txt'.format(tag.replace("_","")), 'w', encoding='utf-8') as f:
        for value in set(values):
            f.write(value+'\n')




# with open('patterns.txt', 'w', encoding='utf-8') as f:
#     for pattern, count in reversed(sorted(all_patterns.items(), key=lambda x: x[1])):
#         f.write('{}\t{}\n'.format(pattern, count))

# print('Total postive queries found: {} | Total patterns: {}'.format(pos_queries_cnt, len(all_patterns)))

# for tag, values in all_tags.items():
#     with open('./placeholder_tags/{}.txt'.format(tag.replace("_","")), 'w', encoding='utf-8') as f:
#         for value in set(values):
#             f.write(value+'\n')
        

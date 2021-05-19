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
from shutil import copyfile
from tqdm import tqdm
import math
from pathlib import Path



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





#### parameter ###
#                # 
##################
domain_extention = 'maxent.output.txt'
intent_extention = 'multiclass.output.txt'
qas_output_file_prefix = 'E:/fileAnswer_data_synthesis/CMF_training/msb_bing_com_mining/data_preprocess/output_with_qas_result/'
merge_qas_output_file_prefix = 'E:/fileAnswer_data_synthesis/CMF_training/msb_bing_com_mining/data_preprocess/merge_with_qas_result/'

#### parameter ###
#                # 
##################


# change to another place
#'E:/mdm_data_analysis/web_fallback_profiling/*'
#E:\mdm_data_analysis\web_fallback_profiling\MDM
#data_files = glob('./original_parse_data/*.tsv')
# list of files
seperate_data_files = glob('E:/fileAnswer_data_synthesis/CMF_training/msb_bing_com_mining/data_preprocess/output/*')
#qas_data_files = glob('E:/fileAnswer_data_synthesis/CMF_training/msb_bing_com_mining/data_preprocess/output_with_qas_result/*')
qas_data_files_set = set(glob(qas_output_file_prefix+'*'))




fallback_df = None
df_list = [] 

for file in seperate_data_files[:]:

    #print("file name {}".format(file))


    #file_wo_extention = file.rsplit('.', 1)[0]


    file_base=os.path.basename(file)
    file_base_wo_extention = os.path.splitext(file_base)[0]


    # for debug
    #print("file name wo extention {}".format(file_base_wo_extention))

    domain_file = qas_output_file_prefix+file_base_wo_extention+'.'+domain_extention
    intent_file = qas_output_file_prefix+file_base_wo_extention+'.'+intent_extention

    # for deubg
    #print("domain_file {}".format(domain_file))

    if os.path.exists(domain_file) and os.path.exists(intent_file):
        # for debug
        print("domain_file found {}".format(domain_file))
        print("intent_file found {}".format(intent_file))

        df = None
        df_domain = None
        df_intent = None

        try:
            # header already no need to append header
            df = pd.read_csv(file, sep='\t', encoding="utf-8", keep_default_na=False,
                #names=["Query", "ImpressionCount", "Domain", "DomainCount", "DomainClicks"],
                dtype={
                'Query': str, 'ImpressionCount': object, 'Domain': object, 'DomainCount': object, 'DomainClicks': object})


            # need to append header
            df_domain = pd.read_csv(domain_file, sep='\t', encoding="utf-8", keep_default_na=False,
                names=["Query", "FilesDomainScore"],
                dtype={
                'Query': str, 'FilesDomainScore': float})

            # apped only two header does not work
            #df_intent = pd.read_csv(intent_file, sep='\t', encoding="utf-8", keep_default_na=False,
            #    names=["Query", "TopIntent"],
            #    dtype={
            #    'Query': str, 'TopIntent': str})

            # no header
            #df_intent = pd.read_csv(intent_file, sep='\t', encoding="utf-8", keep_default_na=False)
            df_intent = pd.read_csv(intent_file, sep='\t', encoding="utf-8", keep_default_na=False, header=None)
        except pd.errors.ParserError:
            print('igonre files: {} due to miss necessary columns, check in the future'.format(file))
            continue




        
        # due to tool restration
        # domain : remove fake old header and last row
        # rename index as well since fake old header is removed
        # <bug>? if len i zero, here will break, test in the fufutre
        df_domain = df_domain.iloc[1:len(df_domain.index)-1]
        df_domain.reset_index(inplace = True)
        
      
        # domain, remove last row
        #df_domain = df_domain.iloc[0:len(df_domain.index)-1]




        # for debug 
        # output intent row by row
        #for i, row in df_intent.iterrows():
        #    print('intent row {}'.format(str(row['TopIntent']).lower()))





        # select only part of columns in domain
        df_domain = df_domain[['FilesDomainScore']]



        # met 1: intent column name does not worl
        # select intent column from intent without header
        # then append TopIntent header
        '''
        df_intent = df_intent.iloc[:, 1]
        print('doman intent before data {}'.format(df_intent.head()))
        df_intent.columns = ['TopIntent']
        print('doman intent data {}'.format(df_intent.head()))
        '''

        # met2: append column name again
        # but remove extra index column appended
        # ? not sure why removing here does work
        '''
        df_intent = df_intent.iloc[:, 1]
        df_intent=pd.DataFrame(df_intent.values, columns = ['TopIntent'])   
        df_intent = df_intent[['TopIntent']]
        '''

        # met2: append column name again
        # extra index column will be drop at the end
        df_intent = df_intent.iloc[:, 1]
        df_intent=pd.DataFrame(df_intent.values, columns = ['TopIntent'])   



        # intent : remove remove fake old header
        # rename index as well since fake old header is removed
        # <bug>? if len i zero, here will break, test in the fufutre
        df_intent = df_intent.iloc[1:len(df_intent.index)]
        df_intent.reset_index(inplace = True)
        #print('doman intent data {}'.format(df_intent.head()))


        # for debug
        #print('original len {}'.format(len(df.index))) 
        #print('doman len {}'.format(len(df_domain.index)))
        #print('intent len {}'.format(len(df_intent.index)))

        # for debug , outpu head
        #print('original head data {}'.format(df.head()))
        #print('doman head data {}'.format(df_domain.head()))
        #print('doman intent data {}'.format(df_intent.head()))



        # assertion checking
        # since some are inconsistent
        # not using assertion , using if-else
        #assert (len(df_domain.index) == len(df.index))
        #assert (len(df_intent.index) == len(df.index))
        if len(df_domain.index) != len(df.index) or len(df_intent.index) != len(df.index):
            print('igonre files: {} due to query unrecognized by as tool, check in the future'.format(file))
            continue



        merge_df = pd.concat([df, df_domain,df_intent], axis=1)
        

        # due to intent effect, need to remove extra index column
        merge_df.drop(['index'], axis='columns', inplace=True)


        #filter by domain score
        merge_df = merge_df.loc[(merge_df['FilesDomainScore'] > 0.35)]
        #filter by intent
        merge_df = merge_df.loc[(merge_df['TopIntent'] == 'file_search')] 

        merge_df.to_csv(merge_qas_output_file_prefix+file_base_wo_extention+'_merged.tsv', index=None, sep='\t')


    #if fallback_df is None:
    #    fallback_df = df2
    #else:
    #    fallback_df = pd.concat(fallback_df, df2)


    #df_list.append(df2)

'''
fallback_df = pd.concat(df_list)
print("len before duplication {}".format(len(fallback_df)))
#fallback_df = fallback_df.sort_values("query", inplace=True)
#fallback_df.sort_values("query", inplace=True)
#fallback_df = fallback_df.drop_duplicates()
#print("len after duplication {}".format(len(fallback_df)))
'''



'''
web_df = pd.concat(df_list)
print("len before duplication {}".format(len(web_df)))
#web_df = web_df.sort_values("query", inplace=True)
#web_df.sort_values("query", inplace=True)

# ? not sure why deduplication is different from web
#web_df = web_df.drop_duplicates()
#print("len after duplication {}".format(len(web_df)))


#for i, row in fallback_df.iterrows():




print('fallback_df head data {}'.format(fallback_df.head()))
print('web_df head data {}'.format(web_df.head()))
merge_df = pd.concat([fallback_df, web_df], axis=1)
print("merge_df before duplication {}".format(len(merge_df)))


print('merge_df top head data {}'.format(merge_df.head()))

print('merge_df header {}'.format(merge_df.columns))


target_row = merge_df.loc[(merge_df['score'] > merge_df['score1'])]

filter_row = merge_df.loc[(merge_df['score'] <= merge_df['score1'])]

print("target_row row {}".format(len(target_row)))
print('target_row top head data {}'.format(target_row.head(300)))

print("filter_row row {}".format(len(filter_row)))
print('filter_row top head data {}'.format(filter_row.head(300)))


# coment this sorry since wanting to seperate
#filter_row.to_csv('E:/mdm_data_analysis/web_fallback_profiling/'+'web_still_win_'+'_updated.tsv', index=None, sep='\t')
#target_row.to_csv('E:/mdm_data_analysis/web_fallback_profiling/'+'web_potential_undertrigger_'+'_updated.tsv', index=None, sep='\t')



target_row.replace(to_replace='&', value='&amp;', inplace=True)



modulo = 60000
# seperate target _row into multiple files
iterCnt = math.ceil(len(target_row) / modulo)

preBound = 0 
for i in range(iterCnt):
    bound = min((i+1)* modulo, len(target_row))
    sub_target_row = target_row.iloc[preBound:bound]




    #sub_target_row.replace(to_replace=r'&', value='&amp;', regex=True)
    
    sub_target_row["query"] = sub_target_row["query"].str.replace("&", "&amp;")
    sub_target_row["query1"] = sub_target_row["query1"].str.replace("&", "&amp;")

    sub_target_row.to_csv('E:/mdm_data_analysis/web_fallback_profiling/'+'web_potential_undertrigger_'+str(i)+'_updated.tsv', index=None, sep='\t')

    preBound = bound
'''





# for testing replacement
# this version work
'''
test_df = pd.read_csv('E:/mdm_data_analysis/web_fallback_profiling/replacement_test.tsv', sep='\t', encoding="utf-8", keep_default_na=False, dtype={
    'query': str, 'score': float, 'query1': str, 'score1': float})

print('test head data {}'.format(test_df.head()))

test_df["query"] = test_df["query"].str.replace("&", "&amp;")
test_df["query1"] = test_df["query1"].str.replace("&", "&amp;")

print('test head data after replacement {}'.format(test_df.head()))
'''


## deuplication pattern
#all_patterns = defaultdict(int)
## deduplication tag (slot value)
#all_tags = defaultdict(list)

'''
filebasename = 'MDM_TrainSet_01202021v1.tsv'
#target = pd.read_csv('target.tsv', sep='\t', encoding="utf-8")
target = pd.read_csv(filebasename, sep='\t', encoding="utf-8", keep_default_na=False, dtype={
    #'MessageText':str, 'JudgedDomain':str,   'JudgedIntent':str, 'JudgedConstraints':str,
    'MessageId': object, 'Frequency': object, 'ConversationContext': object, 'SelectionIgnore': object})


update_df = target.copy()



#patterns = set()  # pattern
#patterns = {}  # pattern
        # for duplicate pattern, frequency_offset will store the maximum one

#patterns_ConversationContext = {}  # pattern: userfilenames json
#patterns_domain = {}  # pattern: 
#patterns_intent = {}  # pattern:
#patterns_annotation = {}  # pattern: 

'''


'''
patterns = {}  # pattern : related_querycontent, using ConversationContext to identify

default_ConversationContext = '#'

# prepare wrong question
for file in data_files[:]:
    df = pd.read_csv(file, sep='\t', encoding="utf-8", keep_default_na=False, dtype={
    #'MessageText':str, 'JudgedDomain':str,   'JudgedIntent':str, 'JudgedConstraints':str,
    'MessageId': object, 'Frequency': object, 'ConversationContext': object, 'SelectionIgnore': object, 'MaxAnnotations': object})
    #'MessageId': object, 'Frequency': object, 'ConversationContext': str, 'SelectionIgnore': object, 'MaxAnnotations': object})

    # big case / small case are different
    for i, row in df.iterrows():
        if row['MessageText'] not in patterns:
            patterns[row['MessageText']] = set()

        # store rest of content as string

        # for debug 
        #print("type1 : {}".format(type(row['JudgedDomain'])))
        #print("type2 : {}".format(type(row['JudgedIntent'])))
        #print("type3 : {}".format(type(row['JudgedIntent'])))
        #print("type4 : {}".format(type(row['ConversationContext'])))
        #print("type5 : {}".format(type(file)))

        if len(str(row['ConversationContext'])) > 0:  
            patterns[row['MessageText']].add(
            str(row['JudgedDomain'])+'\t'+str(row['JudgedIntent']) +'\t'+str(row['JudgedConstraints'])+'\t'+ str(row['ConversationContext']) +'\t'+ str(file))
        else:
            # add dummy one ;'#'
            patterns[row['MessageText']].add(
            str(row['JudgedDomain'])+'\t'+str(row['JudgedIntent']) +'\t'+str(row['JudgedConstraints'])+'\t'+ default_ConversationContext +'\t'+ str(file))

        #patterns_domain[row['MessageText']] = row['JudgedDomain']
        #patterns_intent[row['MessageText']] = row['JudgedIntent']
        #patterns_annotation[row['MessageText']] = row['JudgedConstraints']
        #patterns_ConversationContext[row['MessageText']] = = row['ConversationContext']


# data frame length
# duplcate will not be remove
# for debug
print('update_df size {}'.format(update_df.shape))
print('update_df top head data {}'.format(update_df.head()))

# modified dataset
for i, row in tqdm(update_df.iterrows()):
    pattern = str(row['MessageText'])
    pattern_domain = str(row['JudgedDomain'])
    pattern_intent = str(row['JudgedIntent'])
    pattern_annotation = str(row['JudgedConstraints'])

    if len(str(row['ConversationContext'])) > 0:  
        pattern_ConversationContext = str(row['ConversationContext'])
    else:
        pattern_ConversationContext = default_ConversationContext

    # for deubg 
    #print("pattern {} with context {} before checking".format(pattern,pattern_ConversationContext))

    # skip key does not exist
    if pattern not in patterns:
        continue
    # for deubg 
    #print("pattern {} inside with context {}".format(pattern, pattern_ConversationContext))

    patterns_related_infos = patterns[pattern]

    for patterns_related_info in patterns_related_infos:

        strs = patterns_related_info.split("\t")


        assert (len(strs) >= 5)
        #assert len(strs) == 5, print("strs length unmatch {}".format(len(strs)))

        # for debug
        #print("str len {}".format(len(strs)))
        #print("with context {}".format(strs[0]))
        #print("with context {}".format(strs[1]))
        #print("with context {}".format(strs[2]))
        #print("with context {}".format(strs[3]))
        #print("with context {}".format(strs[4]))

        # using conversationContext to decide whether the same query or not
        if strs[3] == pattern_ConversationContext:
            print("file: {} modifiy query: {} with: context: {}".format(strs[4], pattern, strs[3]))


            # update order
            #str(row['JudgedDomain'])+'\t'+str(row['JudgedIntent']) +'\t'+str(row['JudgedConstraints'])+'\t'+ str(row['ConversationContext'] '\t'+ file))
            row['JudgedDomain'] = strs[0]
            row['JudgedIntent'] = strs[1]
            row['JudgedConstraints'] = strs[2]
            # no need to update this since it is the same
            #row['ConversationContext'] = row['JudgedConstraints']


update_df.to_csv(filebasename+'_updated.tsv', index=None, sep='\t')

'''


'''
for i, row in update_df.iterrows():



    target_row = update_df.loc[(update_df['ConversationId'] == row['ConversationId']) & (update_df['MessageId'] == row['MessageId'])]
            #target_row = update_df.loc[update_df['ConversationId'] == row['ConversationId'] & 
            #update_df['MessageId'] == row['MessageId']]
            if (len(target_row.index) >0):

                #for debug 
                #if filebasename == 'Reporting_TVS_AllDomain_SR_20191201-20191231_7k.tsv':
                #    print('replace convid {}'.format(row['ConversationId']))
                #    print('replace MessageText {}'.format(row['MessageText']))
                #    print('replace JudgedDomain {}'.format(row['JudgedDomain']))


                #print('target_row {}'.format(target_row))
                #print('convid {}'.format(row['ConversationId']))
                #print('mesvid {}'.format(row['MessageId']))
                target_row['JudgedDomain'] = row['JudgedDomain']
                target_row['JudgedIntent'] = row['JudgedIntent']
                target_row['JudgedConstraints'] = row['JudgedConstraints']
                update_df.update(target_row)

'''


'''
for pattern in patterns:
    domain = patterns_domain[pattern]
    intent = patterns_intent[pattern]
    annotation = patterns_annotation[pattern]
    conversationContext = patterns_ConversationContext[pattern]
'''


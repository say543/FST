from glob import glob
import pandas as pd
import re
from pprint import pprint
import string
from nltk.tokenize import word_tokenize
from nltk.util  import ngrams
from nltk.util  import bigrams
#https://stackoverflow.com/questions/17531684/n-grams-in-python-four-five-six-grams
#https://stackoverflow.com/questions/17531684/n-grams-in-python-four-five-six-grams
from collections import defaultdict
import json
import os
import csv
from shutil import copyfile
from tqdm import tqdm
import codecs;



class DataAugmentation(object):
    
    def __init__(self):
        self.tags = []
        self.additonal_patterns = {}       
        
    def load_tags(self, tags):
        self.tags = tags
        
    def get_similar_patterns(self, pattern):
        pattern_tokens = pattern.split()
        similar_patterns = []
        # TODO: Add augmentation methods
        if "file" in pattern and 'the' not in pattern_tokens and 'a' not in pattern_tokens:
            similar_patterns.append(pattern.replace(" file ", " a file ", 1))
            similar_patterns.append(pattern.replace(" file ", " the file ", 1))
        if "document" in pattern and 'the' not in pattern_tokens and 'a' not in pattern_tokens:
            similar_patterns.append(pattern.replace(" document ", " a document ", 1))
            similar_patterns.append(pattern.replace(" document ", " the document ", 1))
        #similar_patterns.append(pattern)
        return similar_patterns



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



def remove_verb_excluded(verbs, verbs_excluded):
    verbs_after_filter = verbs.copy()

    # ? in the futrue , consider past term or present term in the future
    # remove verb from excluded list
    for verb_excluded in verbs_excluded:

        verb_remove_list = []
        for verb in verbs_after_filter:

            # for debug 
            #if verb == 'find':
            #    print('should remove: {}...'.format(verb))


            if verb.lower().find(verb_excluded.lower()) != -1:
                # for deubg
                #print('remove: {}...'.format(verb))
                verb_remove_list.append(verb)

        # remove from verb
        for verb in verb_remove_list:
            verbs_after_filter.remove(verb) 

    return verbs_after_filter    

# function to store the content
def get_ngrams_per_query(text, n):
    n_grams = ngrams(word_tokenize(text), n)
    return [ ' '.join(grams) for grams in n_grams]


def get_ngrams(data_files, xgram_para = 2):

    xgram_query = {}
    xgram_cnt = {}
    xgram_file = {}

    df_list = []


    # prepare wrong question
    for file in data_files[:]:
        df = pd.read_csv(file, sep='\t', encoding="utf-8", keep_default_na=False, dtype={
        #'MessageText':str, 'JudgedDomain':str,   'JudgedIntent':str, 'JudgedConstraints':str,
        #'Query': object, 'ImpressionCount': object, 'Domain': object, 'DomainCount': object, 'DomainClicks': object, 'FilesDomainScore': object, 'TopIntent': object, 'SlotOutput': object})
        #'Query': str, 'ImpressionCount': object, 'Domain': object, 'DomainCount': object, 'DomainClicks': object, 'FilesDomainScore': float, 'TopIntent': str, 'SlotOutput': object})
        'Query': str, 'ImpressionCount': object, 'Domain': object, 'DomainCount': object, 'DomainClicks': object, 'FilesDomainScore': float, 'TopIntent': str})


        print("processing file for xgram_para {}... {}".format(xgram_para, file))

        # drop unncessary columns
        
        # deduplication
        df = df.drop_duplicates()

        #filter by domain score
        df = df.loc[(df['FilesDomainScore'] > 0.35)]

        #filter by intent
        df = df.loc[(df['TopIntent'] == 'file_search')]



        # drop columns other than 'Query'
        df.drop(['ImpressionCount', 'Domain', 'DomainClicks', 'FilesDomainScore', 'TopIntent'], axis='columns', inplace=True)



        df_list.append(df)


        # replace special character
        # done in originla preprocessing
        #df["Query"] = df["Query"].str.replace("&", "&amp;")

        for i, row in df.iterrows():
            query = str(row['Query']).lower()

            # for debug
            #print("query : {}".format(word_tokenize(query)))
        

            #xgrams = ngrams(query, xgram_para)
            #tokens_for_xgrams = ngrams(word_tokenize(query), xgram_para)
            #xgrams = [ ' '.join(grams) for grams in tokens_for_xgrams]
            xgrams = get_ngrams_per_query(query, xgram_para)

            # for debug
            #print("xgrams : {}".format(xgrams))
        
            #xgrams = bigrams(query)

            for gram in xgrams:

                # for debug
                #print("gram : {}".format(gram))
                #print("gram size : {}".format(gram))


                if gram in xgram_query:
                    xgram_query[gram].add(row['Query'])
                    #if gram == 'files azure':
                    #    print("gram {} add: {}".format(gram, row['Query']))

                    #print("gram {} add: {}".format(gram, row['Query']))
                else:
                    xgram_query[gram] = set()
                    xgram_query[gram].add(row['Query'])
            
                #xgram_query[gram] = row['Query']
                xgram_file[gram] = file

                if gram in xgram_cnt:
                    xgram_cnt[gram] = xgram_cnt[gram] +1
                else:
                    xgram_cnt[gram] = 1

    # for debug
    #print("query for files azure: {}".format(xgram_query['files azure']))


    return xgram_query, xgram_cnt , xgram_file, df_list


# function to filter
def ngrams_compound_heurictic_cat1(xgram, filetypeIncludeBoost, verbs):
    xgram_compound_after_filter = {}

    for key, value in xgram.items():

        # filter out non 2 gram 
        if len(word_tokenize(key)) != 2:
            print('input token is not 2 gram, ignore {}'.format(key))
            continue

        
        tokens = word_tokenize(key)

        # category 1
        # filetypeIncludeBoost + non-verb
        if tokens[0].lower() in filetypeIncludeBoost and tokens[1].lower() not in verbs:
            isSubString = False
            for verb in verbs:
                if verb.lower().find(tokens[1].lower()) != -1:
                    isSubString = True
                    break

            if isSubString is False:
                xgram_compound_after_filter[key] = value
            
    return xgram_compound_after_filter

def ngrams_compound_heurictic_cat2(xgram, filetypeIncludeBoost, verbs, verbs_excluded):
    xgram_compound_after_filter = {}

    verbs_after_filter = remove_verb_excluded(verbs, verbs_excluded)

    '''
    verbs_after_filter = verbs.copy()

    # ? in the futrue , consider past term or present term in the future
    # remove verb from excluded list
    for verb_excluded in verbs_excluded:

        verb_remove_list = []
        for verb in verbs_after_filter:
            # if inside
            if verb.lower().find(verb_excluded.lower()) != -1:
                verb_remove_list.append(verb)

        # remove from verb
        for verb in verb_remove_list:
            verbs_after_filter.remove(verb)
    '''

    for key, value in xgram.items():

        # filter out non 2 gram 
        if len(word_tokenize(key)) != 2:
            print('input token is not 2 gram, ignore {}'.format(key))
            continue

        
        tokens = word_tokenize(key)





        # category 2
        # non-verb + filetypeIncludeBoost
        # token0: Exclude valid verb
        # token0: Exclude filetypeIncludeBoost as well to mkae list smaller
        if tokens[0].lower() not in verbs_after_filter and \
           tokens[0].lower() not in filetypeIncludeBoost and \
            tokens[1].lower() in filetypeIncludeBoost :
            isSubString = False
            for verb in verbs_after_filter:
                if verb.lower().find(tokens[0].lower()) != -1:
                    isSubString = True
                    break

            if isSubString is False:

                #for debug
                #print('input token is added {}/{}'.format(tokens[0], tokens[1])) 

                xgram_compound_after_filter[key] = value
            
    return xgram_compound_after_filter


def pattern_heurictic_cat1(df_list, filetypeIncludeBoost, verbs):
    xgram_compound_after_filter = {}

    # concat and deduplciation
    df = pd.concat(df_list)
    df = df.drop_duplicates()


    df_res_dict = {}


    #debug  for speead up
    #filetypeIncludeBoost = set([
    #    'excel',
    #    'excels'
    #])


    for filetypeIncludeBoostEle in filetypeIncludeBoost:
        for verb in verbs:

            # not sure why has verb empty 
            # add empty filter 
            if len(verb) == 0:
                continue

            reg = filetypeIncludeBoostEle.lower() + '.*' + ' '+verb +' '


            # for deubg
            print('matching reg: {}...'.format(reg))

            #insensitve match
            df_temp = df[df['Query'].str.match(reg, case =False)== True]
            df_temp = df_temp.drop_duplicates()
            df_res_dict[reg] = df_temp

            # for deubg
            print('matching num rows: {}...'.format(len(df_temp)))

    return df_res_dict


def pattern_heurictic_cat2(df_list, filetypeIncludeBoost, verbs, verbs_excluded):
    xgram_compound_after_filter = {}

    verbs_after_filter = remove_verb_excluded(verbs, verbs_excluded)

    # concat and deduplciation
    df = pd.concat(df_list)
    df = df.drop_duplicates()


    df_res_dict = {}


    #debug  for speead up
    #filetypeIncludeBoost = set([
    #    'excel',
    #    'excels'
    #])


    for filetypeIncludeBoostEle in filetypeIncludeBoost:
        for verb in verbs_after_filter:

            # not sure why has verb empty 
            # add empty filter 
            if len(verb) == 0:
                continue

            reg = verb + '.*' + ' '+filetypeIncludeBoostEle.lower() +' '


            # for deubg
            print('matching reg: {}...'.format(reg))

            #insensitve match
            df_temp = df[df['Query'].str.match(reg, case =False)== True]
            df_temp = df_temp.drop_duplicates()
            df_res_dict[reg] = df_temp

            # for deubg
            print('matching num rows: {}...'.format(len(df_temp)))

    return df_res_dict









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


neg_queires = set()

with open("pattern_cat2.tsv", 'r', encoding='utf-8') as fin:
    for line in fin:
        line = line.strip();
        if not line:
            continue;
        linestrs = line.split("\t");
        # make sure it at least has
        # Query	ExternalFeature	Weight	Intent	Domain	Slot
        if len(linestrs) < 2:
            continue;

        pattern = linestrs[0]
        
        patternHead = (pattern.split("."))[0]

        queryList = linestrs[1][1:len(linestrs[1])-1]
        #print('pattern {} , query : {} ...'.format(pattern, queryList))
        queriesList = queryList.split(',')


        #print('pattern {} , query : {} ...'.format(pattern, linestrs[1]))

        # convert set to string
        #https://www.javaexercise.com/python/python-convert-set-to-string


        #print('str_val : {}'.format(linestrs[1]))


        #queryset = 'set(%s)' % linestrs[1]


        #print(type(queryset))

        #str_val = "\t".join(map(str,linestrs[1]))
        #print('str_val : {}'.format(str_val))

        for query in queriesList:
            # remove extra space 
            query = query.strip()
            # remove quote
            query = query[1:len(query)-1]

            # for deubg 
            #print('patternhead: {} , query: {}'.format(patternHead, query))

            if query.lower().find((' '+ patternHead.lower()+' ').lower()) != -1:
                neg_queires.add(query)
            elif query.lower().startswith(patternHead.lower()+' '):
                neg_queires.add(query)
            else:
                print('filter patternhead: {} , query: {}'.format(patternHead, query))




with open("pattern_cat2_filter.tsv", 'w', encoding='utf-8') as fout:
    fout.write('\t'.join(['TurnNumber', 'PreviousTurnIntent', 'query','domain','PreviousTurnDomain', \
    'TaskFrameStatus', 'TaskFrameEntityStates', 'TaskFrameGUID', 'SpeechPeopleDisambiguationGrammarMatches']) + '\n')


    for query in tqdm(neg_queires):
        if len(query) > 0:

            #ouput list 
            #fout.write(str(key)+'\t'+str(df['Query'].tolist())+'\n')
            # output set'
            fout.write('0'+'\t\t'+str(query)+'\t'+'web'+'\t\t\t\t\t\t\t'+'pattern_cat2.tsv'+'\n')



'''
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


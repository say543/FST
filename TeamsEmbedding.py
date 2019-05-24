
import pandas as pd
import math
import re
import sys
import glob
import numpy as np
import ast

from bert_serving.client import BertClient

mediaTitleRelated = ['song_name', 'album_name', 'movie_name', 'tv_name'];
timeRelated = ['start_time', 'start_date'];


def load_luna_to_carina(filename, generateSentenceEmbedding=False):
    # create a local bert client
    if generateSentenceEmbedding==True:
        bc = BertClient()


    print('Loading ' + filename)
    
    xml_pattern = re.compile('(<.*?key=\"+(.+?)\"+.*?>(.*?)<\/.*?>)')
    slot_validation_pattern = re.compile('^[a-zA-Z_.]+$')
    src = pd.read_csv(filename, sep='\t', low_memory=False)
    
    bad_lines = []
    
    entries = []    
    for index, row in src.iterrows():
        annotation = row['JudgedTaskunderstandingMsgannotation']
        domain = row['judged_domain']
        
        if not domain or str(domain) == 'nan':
            continue;

        """
        if domain.lower() != 'mediacontrol':
            continue;
		"""
        
        try:         
            matches = re.findall(xml_pattern, annotation)
            for tag, slot, text in matches:
                slot_stripped = slot.strip();

                if slot_stripped in mediaTitleRelated:
                    slot_stripped = 'media_title';
                elif slot_stripped in timeRelated:
                    slot_stripped = 'time';
                    
                if not re.match(slot_validation_pattern, slot_stripped):
                    bad_lines.append(str(index) + ': ' + annotation)
                    continue
                
                replacement = text
                replacement = '<{0}> {1} </{2}>'.format(slot_stripped, text, slot_stripped)
                annotation = annotation.replace(tag, replacement)
            
            intent = row['JudgedTaskunderstandingIntent']


            query = [row['MessageText']]
            if generateSentenceEmbedding==True:
                #sentenceEmbedding = bc.encode(query)
                #sentenceEmbedding = bc.encode(query).tolist()
                sentenceEmbedding = bc.encode(query).tolist()[0]

                # for debug 
                #print(f'sentenceEmbedding: {sentenceEmbedding.shape}')
                entry = {
                    'id': 0,
                    'query': row['MessageText'],
                    'intent': intent,
                    'domain': domain.lower(),
                    'QueryXml': annotation,
                    'SentenceEmbedding':  sentenceEmbedding
                    }
            else:
                entry = {
                    'id': 0,
                    'query': row['MessageText'],
                    'intent': intent,
                    'domain': domain.lower(),
                    'QueryXml': annotation
                    }

            entries.append(entry)
        except:
            print("skipping " + row['MessageText'])
         
    if bad_lines:
        for l in bad_lines:
            print(l)
##        raise Exception('Bad slots found. See output above')
        
    #return pd.DataFrame(entries, columns=['QueryId', 'RawQuery', 'Intent', 'Domain', 'SlotXML', 'SentenceEmbedding'])
    if generateSentenceEmbedding==True:
        return pd.DataFrame(entries, columns=['id', 'query', 'intent', 'domain', 'QueryXml', 'SentenceEmbedding'])
    else:
        return pd.DataFrame(entries, columns=['id', 'query', 'intent', 'domain', 'QueryXml'])

def load_carina(filename, generateSentenceEmbedding=False):
    # create a local bert client
    if generateSentenceEmbedding==True:
        bc = BertClient()


    print('Loading ' + filename)
    
    xml_pattern = re.compile('(<.*?key=\"+(.+?)\"+.*?>(.*?)<\/.*?>)')
    slot_validation_pattern = re.compile('^[a-zA-Z_.]+$')
    src = pd.read_csv(filename, sep='\t', low_memory=False)
    
    bad_lines = []
    
    entries = []    
    for index, row in src.iterrows():
        try:         
            intent = row['intent']

            query = [row['query']]
            if generateSentenceEmbedding==True:
                #sentenceEmbedding = bc.encode(query)
                #sentenceEmbedding = bc.encode(query).tolist()
                sentenceEmbedding = bc.encode(query).tolist()[0]
                # for debug 
                #print(f'sentenceEmbedding: {sentenceEmbedding.shape}')
                entry = {
                    'id': 0,
                    'query': row['query'],
                    'intent': row['intent'],
                    'domain': row['domain'].lower(),
                    'QueryXml': row['QueryXml'],
                    'SentenceEmbedding':  sentenceEmbedding
                    }
            else:
                entry = {
                    'id': 0,
                    'query': row['query'],
                    'intent': row['intent'],
                    'domain': row['domain'].lower(),
                    'QueryXml': row['QueryXml'],
                    }

            entries.append(entry)
        except:
            print("skipping " + row['MessageText'])
         
    if bad_lines:
        for l in bad_lines:
            print(l)
##        raise Exception('Bad slots found. See output above')
        
    #return pd.DataFrame(entries, columns=['QueryId', 'RawQuery', 'Intent', 'Domain', 'SlotXML', 'SentenceEmbedding'])
    if generateSentenceEmbedding==True:
        return pd.DataFrame(entries, columns=['id', 'query', 'intent', 'domain', 'QueryXml', 'SentenceEmbedding'])
    else:
        return pd.DataFrame(entries, columns=['id', 'query', 'intent', 'domain', 'QueryXml'])



def updateFileType(df):
    for index, row in df.iterrows():


        # hard code rule
        # no replace since incomplete
        # Show my presentation about
        # Share the presentation functionally in


        # deck_name might be affected by this routine...
        # those three need special care
        # Present the meeting the deck
        # Present the teams meeting intelligence deck
        # Present the teams meeting intelligence stack

        # deck_location might be affected



        # need to fix it
        # Present the teams meeting intelligence stack
        # Present the meeting the deck
        # Present the teams meeting intelligence deck
        # present the deck i was working on  (intent in consistency so update will not happen)



        if df.at[index,'intent'] == 'start_presenting' or df.at[index,'intent'] == 'stop_presenting' or df.at[index,'intent'] == 'goto_slide':

            # replace <deck_name> , </deck_name>
            newQueryXml = df.at[index,'QueryXml'].replace('<deck_name>', '<file_title>')
            newQueryXml = newQueryXml.replace('</deck_name>', '</file_title>')

            # replace plural terms
            newQueryXml = newQueryXml.replace('presentations', 'PLURAL_ps')
            newQueryXml = newQueryXml.replace('decks', 'PLURAL_ds')


            newQueryXml = df.at[index,'QueryXml'].replace('presentation', '<file_filetype> presentation </file_filetype>')
            newQueryXml = newQueryXml.replace('deck', '<file_filetype> deck </file_filetype>')
            
            #prevent duplicate replacement
            newQueryXml = newQueryXml.replace('<file_filetype><file_filetype>', '<file_filetype>')
            newQueryXml = newQueryXml.replace('</file_filetype></file_filetype>', '</file_filetype>')
            newQueryXml = newQueryXml.replace('<file_filetype> <file_filetype>', '<file_filetype>')
            newQueryXml = newQueryXml.replace('</file_filetype> </file_filetype>', '</file_filetype>')


            # replace back plura terms
            newQueryXml = newQueryXml.replace('PLURAL_ps', 'presentations')
            newQueryXml = newQueryXml.replace('PLURAL_ds', 'decks')

            # deprecate usage
            #df.set_value(index, 'QueryXml', newQueryXml)
            df.at[index,'QueryXml'] = newQueryXml
    return df
            
    

def calculate_topk_similarity(df_golden, df_tune):

    '''   
    # query comparison for deubgging

    #golden_sentenceEmbeddings = np.array(df_golden['SentenceEmbedding'])
    #tune_SentenceEmbeddings = np.array(df_tune['SentenceEmbedding'])
    
    #golden_sentenceEmbeddings = np.array(ast.literal_eval(df_golden['SentenceEmbedding'][0]))
    #tune_SentenceEmbeddings = np.array(ast.literal_eval(df_tune['SentenceEmbedding'][0]))
    #golden_sentenceEmbeddings = ast.literal_eval(df_golden['SentenceEmbedding'][0])
    #tune_SentenceEmbeddings = ast.literal_eval(df_tune['SentenceEmbedding'][0])

    #golden_sentenceEmbeddings = df_golden['SentenceEmbedding']
    #tune_SentenceEmbeddings = df_tune['SentenceEmbedding']

    golden_sentenceEmbeddings = [ row[1]   for index, row in df_golden['SentenceEmbedding']]
    #tune_SentenceEmbeddings = [ row[1]   for index, row in df_tune['SentenceEmbedding']]

    #golden_sentenceEmbeddings = [ row[1]   for index, row in df_golden['SentenceEmbedding']]
    #tune_SentenceEmbeddings = [ row[1]   for index, row in df_tune['SentenceEmbedding']]

    #golden_sentenceEmbeddings = df_golden['SentenceEmbedding'].tolist()
    #tune_SentenceEmbeddings = df_tune['SentenceEmbedding'].tolist()


    #print(f"type: {type(golden_sentenceEmbeddings)}")
    #print(f"type: {type(tune_SentenceEmbeddings)}")

    #print(f"len: {len(golden_sentenceEmbeddings)}")
    #print(f"len: {len(tune_SentenceEmbeddings)}")

    #print(f"len: {len(golden_sentenceEmbeddings[0])}")
    #print(f"len: {len(tune_SentenceEmbeddings[0])}")

    print (type(golden_sentenceEmbeddings[0]))
    '''

    #golden_sentenceEmbeddings = np.array([ ast.literal_eval(df_golden['SentenceEmbedding'][i]) for i in range(len(df_golden['SentenceEmbedding']))])
    #tune_SentenceEmbeddings = np.array([ ast.literal_eval(df_tune['SentenceEmbedding'][i]) for i in range(len(df_tune['SentenceEmbedding']))])

    golden_SentenceEmbeddings = np.array([ ast.literal_eval(v) for i, v in df_golden['SentenceEmbedding'].items()   ])
    tune_SentenceEmbeddings = np.array([ ast.literal_eval(v) for i, v in df_tune['SentenceEmbedding'].items() ])

    #golden_sentenceEmbeddings = [ast.literal_eval(df_golden['SentenceEmbedding'])]
    #tune_SentenceEmbeddings = df_tune['SentenceEmbedding'].values

    print(f'Shape: {golden_SentenceEmbeddings.shape}')
    print(f'Shape: {tune_SentenceEmbeddings.shape}')

    # cosine similarity

    tune_SentenceEmbeddings_distance = np.sqrt(np.sum((tune_SentenceEmbeddings)**2, axis=1))
    golden_SentenceEmbeddings_distance = np.sqrt(np.sum((golden_SentenceEmbeddings)**2, axis=1))
    tune_SentenceEmbeddings_distance_column_vector = tune_SentenceEmbeddings_distance.reshape(tune_SentenceEmbeddings_distance.shape[0], 1)
    golden_SentenceEmbeddings_distance_column_vector = golden_SentenceEmbeddings_distance.reshape(golden_SentenceEmbeddings_distance.shape[0], 1)
    
    sentenceEmbeddings_similiary = np.dot (tune_SentenceEmbeddings, golden_SentenceEmbeddings.T) / np.dot(tune_SentenceEmbeddings_distance_column_vector, golden_SentenceEmbeddings_distance_column_vector.T)


    # for top1
    # debug
    # 
    # equal check will fail
    # becasue there might be two items with the same cosine similairty
    #index1 = np.argmax(sentenceEmbeddings_similiary, axis=1)
    #index2 = (np.argpartition(sentenceEmbeddings_similiary, -1, axis=1)[:, -1:]).flatten()
    #diff = index1 - index2
    #np.where(diff!=0)
    #(array([   11,    25,    26, ..., 52174, 52192, 52233], dtype=int64),)
    # checking 11
    # np.argmax(sentenceEmbeddings_similiary[11,:])
    # 55
    # np.argpartition(sentenceEmbeddings_similiary[11,:], -1)[-1:]
    # array([364], dtype=int64)

    # top k . k > 1
    # here no need flattern since wanto to maintain index row for each tune query
    top_k = 4
    top_k_golden_per_tune = (np.argpartition(sentenceEmbeddings_similiary, -top_k, axis=1)[:, -top_k:])

    # get all similairty value based on top k indexes
    top_k_golden_similarity_value_per_tune = np.array([ sentenceEmbeddings_similiary[i, top_k_golden_per_tune[i, :]]    for i in range(sentenceEmbeddings_similiary.shape[0])])


    #top_k_golden_and_similarity_per_tune = np.dstack((top_k_golden_per_tune, top_k_golden_similarity_value_per_tune))
    top_k_similarity_and_golden_per_tune = np.dstack((top_k_golden_similarity_value_per_tune, top_k_golden_per_tune))

    # sort by similarity decreasing
    top_k_similarity_and_golden_per_tune_sorted = np.array([ np.sort(top_k_similarity_and_golden_per_tune[i], axis=0)[::-1]    for i in range(top_k_similarity_and_golden_per_tune.shape[0])])
    top_k_golden_sorted_per_tune = top_k_similarity_and_golden_per_tune_sorted[:,:,1].astype(int)
    top_k_similarity_sorted_per_tune = top_k_similarity_and_golden_per_tune_sorted[:,:,0]

    # get all golden queries by default order 
    #golden_queries = df_golden['query'].values.reshape((df_golden['query']).shape[0],1)
    #closest_golden_queries_per_tune = golden_queries[top_k_golden_per_tune]
    #closest_golden_queries_per_tune = closest_golden_queries_per_tune.reshape(closest_golden_queries_per_tune.shape[0], closest_golden_queries_per_tune.shape[1])


    # get all golden queries by similarity decreasing order 
    golden_queries = df_golden['query'].values.reshape((df_golden['query']).shape[0],1)
    closest_golden_queries_per_tune = golden_queries[top_k_golden_sorted_per_tune]
    closest_golden_queries_per_tune = closest_golden_queries_per_tune.reshape(closest_golden_queries_per_tune.shape[0], closest_golden_queries_per_tune.shape[1])
    
    return top_k_golden_sorted_per_tune, closest_golden_queries_per_tune, top_k_similarity_sorted_per_tune


def filter_by_negative_examples(df_tune, df_negative_examples, extra_negative_only=False):

    # only index
    #if extra_negative_only:
    #    return df_tune[df_tune['query'].isin(df_negative_examples['query'])]
    #else:
    #    return df_tune[~df_tune['query'].isin(df_negative_examples['query'])]

    # hard copy
    if extra_negative_only:
        return (df_tune[df_tune['query'].isin(df_negative_examples['query'])].copy(deep=True)).reset_index(drop=True)
    else:
        return (df_tune[~df_tune['query'].isin(df_negative_examples['query'])].copy(deep=True)).reset_index(drop=True)



def algoV1(df_tune, closest_golden_queries_per_tune, top_k_golden_sorted_per_tune, top_k_similarity_sorted_per_tune):
    # v1 extract all topk

    # deep copy to append closest queries
    df_tune_with_golen_queries = df_tune.copy(deep=True)

    # remove embedding for saving space
    df_tune_with_golen_queries = df_tune_with_golen_queries.drop('SentenceEmbedding', axis=1)

    # output top k all information
    closest_golden_queries_per_tune_pd_series = pd.Series(map(lambda x: x[:], closest_golden_queries_per_tune))
    top_k_golden_sorted_per_tune_pd_series = pd.Series(map(lambda x: x[:], top_k_golden_sorted_per_tune))
    top_k_similarity_sorted_per_tune_pd_series = pd.Series(map(lambda x: x[:], top_k_similarity_sorted_per_tune))

    df_tune_with_golen_queries['GoldenQuery'] = closest_golden_queries_per_tune_pd_series
    df_tune_with_golen_queries['GoldenQueryIndex'] = top_k_golden_sorted_per_tune_pd_series
    df_tune_with_golen_queries['GoldenQuerySimilarity'] = top_k_similarity_sorted_per_tune_pd_series
    # persistent to file
    df_tune_with_golen_queries.to_csv('E:\\bert-as-a-service\\TeamsEmbedding\\TeamsEmbedding\\top4.tsv', sep='\t', index=None)

    return df_tune_with_golen_queries

def algoV2(df_tune, closest_golden_queries_per_tune, top_k_golden_sorted_per_tune, top_k_similarity_sorted_per_tune):
    # v2 extract the last one of top k 
    # deep copy to append closest queries
    df_tune_with_last_golen_queries = df_tune.copy(deep=True)
    df_tune_with_last_golen_queries = df_tune_with_last_golen_queries.drop('SentenceEmbedding', axis=1)
    last_closest_golden_queries_per_tune_pd_series = pd.Series(map(lambda x: x[3], closest_golden_queries_per_tune))
    last_golden_sorted_per_tune_pd_series = pd.Series(map(lambda x: x[3], top_k_golden_sorted_per_tune))
    last_similarity_sorted_per_tune_pd_series = pd.Series(map(lambda x: x[3], top_k_similarity_sorted_per_tune))

    df_tune_with_last_golen_queries['GoldenQuery'] = last_closest_golden_queries_per_tune_pd_series
    df_tune_with_last_golen_queries['GoldenQueryIndex'] = last_golden_sorted_per_tune_pd_series
    df_tune_with_last_golen_queries['GoldenQuerySimilarity'] = last_similarity_sorted_per_tune_pd_series

    # persistent to file
    df_tune_with_last_golen_queries.to_csv('E:\\bert-as-a-service\\TeamsEmbedding\\TeamsEmbedding\\last_of_top4.tsv', sep='\t', index=None)

    return df_tune_with_last_golen_queries

def algoV3(df_tune, closest_golden_queries_per_tune, top_k_golden_sorted_per_tune, top_k_similarity_sorted_per_tune):
    # v3 extract the first one of top k
    # deep copy to append closest queries
    df_tune_with_first_golen_queries = df_tune.copy(deep=True)
    df_tune_with_first_golen_queries = df_tune_with_first_golen_queries.drop('SentenceEmbedding', axis=1)
    first_closest_golden_queries_per_tune_pd_series = pd.Series(map(lambda x: x[0], closest_golden_queries_per_tune))
    first_golden_sorted_per_tune_pd_series = pd.Series(map(lambda x: x[0], top_k_golden_sorted_per_tune))
    first_similarity_sorted_per_tune_pd_series = pd.Series(map(lambda x: x[0], top_k_similarity_sorted_per_tune))

    df_tune_with_first_golen_queries['GoldenQuery'] = first_closest_golden_queries_per_tune_pd_series
    df_tune_with_first_golen_queries['GoldenQueryIndex'] = first_golden_sorted_per_tune_pd_series
    df_tune_with_first_golen_queries['GoldenQuerySimilarity'] = first_similarity_sorted_per_tune_pd_series

    # persistent to file
    df_tune_with_first_golen_queries.to_csv('E:\\bert-as-a-service\\TeamsEmbedding\\TeamsEmbedding\\first_of_top4.tsv', sep='\t', index=None)

    return df_tune_with_first_golen_queries

if __name__ == "__main__":



    # golden_file_processing , luna format
    golden_file_name="Teams-MustPass_Feb_Golden.tsv" 
    #df_golden = load_luna_to_carina(golden_file_name, generateSentenceEmbedding=True)
    #df_golden.to_csv('.\\' + golden_file_name.replace('.tsv', '-carina.tsv'), sep='\t', index=None)

    # fake_file_processing, carina format
    tune_file_name="Teams_Slot_Training.tsv"
    #df_tune = load_carina(tune_file_name, generateSentenceEmbedding=True)
    #df_tune.to_csv('.\\' + tune_file_name.replace('.tsv', '-carina.tsv'), sep='\t', index=None)

    # negative examples
    # no need embedding
    negative_examples_file_name = "selected_web_queries_negative_examples.tsv"
    df_negative_examples =  pd.read_csv(negative_examples_file_name, sep='\t')

    # read from previous embedding
    print(f"load from file...")
    df_golden = pd.read_csv(golden_file_name.replace('.tsv', '-carina.tsv'), sep='\t') 
    df_tune = pd.read_csv(tune_file_name.replace('.tsv', '-carina.tsv'), sep='\t')
    print(f"load from file done")


    
    # debug
    #df_golden_generate = load_luna_to_carina(golden_file_name)
    #print(f"initial and read-from-file comparasion: {df_golden_generate['query'].equals(df_golden['query'])}")


    # filter by negative samples
    #dt_tune_without_negative_examples = (df_tune[~df_tune['query'].isin(df_negative_examples['query'])].copy(deep=True)).reset_index(drop=True)
    #dt_tune_negative_examples = (df_tune[df_tune['query'].isin(df_negative_examples['query'])].copy(deep=True)).reset_index(drop=True)
    dt_tune_without_negative_examples = filter_by_negative_examples(df_tune, df_negative_examples, extra_negative_only=False)
    dt_tune_negative_examples = filter_by_negative_examples(df_tune, df_negative_examples, extra_negative_only=True)


    # calculate similairty v1
    # use all samples
    #top_k_golden_sorted_per_tune, closest_golden_queries_per_tune, top_k_similarity_sorted_per_tune = calculate_topk_similarity(df_golden, df_tune)
    # v1 extract all topk
    #df_tune_with_golen_queries = algoV1(df_tune, closest_golden_queries_per_tune, top_k_golden_sorted_per_tune, top_k_similarity_sorted_per_tune)

    # v2 extract the last one of top k 
    #df_tune_with_golen_queries = algoV2(df_tune, closest_golden_queries_per_tune, top_k_golden_sorted_per_tune, top_k_similarity_sorted_per_tune)

    # v3 extract the first one of top k 
    #df_tune_with_golen_queries = algoV3(df_tune, closest_golden_queries_per_tune, top_k_golden_sorted_per_tune, top_k_similarity_sorted_per_tune)



    # calculate similairty v2
    # use all positive examples
    top_k_golden_sorted_per_tune, closest_golden_queries_per_tune, top_k_similarity_sorted_per_tune = calculate_topk_similarity(df_golden, dt_tune_without_negative_examples)

    # v1 extract all topk
    #df_tune_with_golen_queries = algoV1(dt_tune_without_negative_examples, closest_golden_queries_per_tune, top_k_golden_sorted_per_tune, top_k_similarity_sorted_per_tune)

    # v2 extract the last one of top k 
    df_tune_with_golen_queries = algoV2(dt_tune_without_negative_examples, closest_golden_queries_per_tune, top_k_golden_sorted_per_tune, top_k_similarity_sorted_per_tune)

    # v3 extract the first one of top k 
    #df_tune_with_golen_queries = algoV3(dt_tune_without_negative_examples, closest_golden_queries_per_tune, top_k_golden_sorted_per_tune, top_k_similarity_sorted_per_tune)


    # merge negative examples back with similairty = 1, golden query = -1
    dt_tune_negative_examples = dt_tune_negative_examples.drop('SentenceEmbedding', axis=1)
    dt_tune_negative_examples['GoldenQuery'] = "NonAvail"
    dt_tune_negative_examples['GoldenQueryIndex'] = -1
    dt_tune_negative_examples['GoldenQuerySimilarity'] = 1.0 # double type
    df_tune_with_golen_queries_with_shuffle = pd.concat([df_tune_with_golen_queries, dt_tune_negative_examples]).reset_index(drop=True)

    df_tune_with_first_golen_queries.to_csv('E:\\bert-as-a-service\\TeamsEmbedding\\TeamsEmbedding\\first_of_top4_with_shuffle.tsv', sep='\t', index=None)


    # debug 
    # cutoff by threshold 0.94
    # cutoff = 0.94
    #df_tune_with_last_golen_queries_similarity_condition_bigger_than = df_tune_with_last_golen_queries['GoldenQuerySimilarity'] > 0.94
    #df_tune_with_last_golen_queries_similarity_bigger_than = df_tune_with_last_golen_queries[df_tune_with_last_golen_queries_similarity_condition_bigger_than]

    #df_tune_with_last_golen_queries_similarity_bigger_than_without_negative_example =  df_tune_with_last_golen_queries_similarity_bigger_than[df_tune_with_last_golen_queries_similarity_bigger_than['query'].isin(df_negative_examples['query'])]




    '''
    # slotFiles = glob.glob("*.tsv");
    slotFiles = glob.glob("Teams-MustPass_Feb_Golden.tsv");
    for slotFile in slotFiles:
        #load luna data and transform to carina
        #df = load_as_carina(slotFile)

        #load luna data, transform to carina, generate sentence embedding
        #df = load_as_carina_and_embedding(slotFile)

        #generate embedding to pickles

        #generate csv with embedding
        #df.to_csv('.\\' + slotFile.replace('.tsv', '-carina.tsv'), sep='\t', index=None)    
    '''



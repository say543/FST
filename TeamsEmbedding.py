
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
                    'domain': domain,
                    'QueryXml': annotation,
                    'SentenceEmbedding':  sentenceEmbedding
                    }
            else:
                entry = {
                    'id': 0,
                    'query': row['MessageText'],
                    'intent': intent,
                    'domain': domain,
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
                    'domain': row['domain'],
                    'QueryXml': row['QueryXml'],
                    'SentenceEmbedding':  sentenceEmbedding
                    }
            else:
                entry = {
                    'id': 0,
                    'query': row['query'],
                    'intent': row['intent'],
                    'domain': row['domain'],
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

    golden_sentenceEmbeddings = np.array([ ast.literal_eval(v) for i, v in df_golden['SentenceEmbedding'].items()   ])
    tune_SentenceEmbeddings = np.array([ ast.literal_eval(v) for i, v in df_tune['SentenceEmbedding'].items() ])

    #golden_sentenceEmbeddings = [ast.literal_eval(df_golden['SentenceEmbedding'])]
    #tune_SentenceEmbeddings = df_tune['SentenceEmbedding'].values

    print(f'Shape: {golden_sentenceEmbeddings.shape}')
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


    # get all golden queries
    golden_queries = df_golden['query'].values.reshape((df_golden['query']).shape[0],1)
    closest_golden_queries_per_tune = golden_queries[top_k_golden_per_tune]
    closest_golden_queries_per_tune = closest_golden_queries_per_tune.reshape(closest_golden_queries_per_tune.shape[0], closest_golden_queries_per_tune.shape[1])
    
    return top_k_golden_per_tune, closest_golden_queries_per_tune

if __name__ == "__main__":



    # golden_file_processing , luna format
    golden_file_name="Teams-MustPass_Feb_Golden.tsv" 
    #df_golden = load_luna_to_carina(golden_file_name, generateSentenceEmbedding=True)
    #df_golden.to_csv('.\\' + golden_file_name.replace('.tsv', '-carina.tsv'), sep='\t', index=None)

    # fake_file_processing, carina format
    tune_file_name="Teams_Slot_Training.tsv"
    #df_tune = load_carina(tune_file_name, generateSentenceEmbedding=True)
    #df_tune.to_csv('.\\' + tune_file_name.replace('.tsv', '-carina.tsv'), sep='\t', index=None)

    # read from previous embedding
    print(f"load from file...")
    df_golden = pd.read_csv(golden_file_name.replace('.tsv', '-carina.tsv'), sep='\t') 
    df_tune = pd.read_csv(tune_file_name.replace('.tsv', '-carina.tsv'), sep='\t')
    print(f"load from file done")

    
    # debug
    #df_golden_generate = load_luna_to_carina(golden_file_name)
    #print(f"initial and read-from-file comparasion: {df_golden_generate['query'].equals(df_golden['query'])}")

    top_k_golden_per_tune, closest_golden_queries_per_tune = calculate_topk_similarity(df_golden, df_tune)

    # deep copy to append closest queries
    df_tune_with_golen_queries = df_tune.copy(deep=True)
    closest_golden_queries_per_tune_pd_series = pd.Series(map(lambda x: x[:], closest_golden_queries_per_tune))
    df_tune_with_golen_queries['GoldenQuery'] = closest_golden_queries_per_tune_pd_series





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



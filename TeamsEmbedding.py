
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

    calculate_topk_similarity(df_golden, df_tune)



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



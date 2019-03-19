
import pandas as pd
import math
import re
import sys
import glob

from bert_serving.client import BertClient

mediaTitleRelated = ['song_name', 'album_name', 'movie_name', 'tv_name'];
timeRelated = ['start_time', 'start_date'];

def load_as_carina(filename):
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
            entry = {
                    'QueryId': 0,
                    'RawQuery': row['MessageText'],
                    'Intent': intent,
                    'Domain': 'mediacontrol',
                    'SlotXML': annotation
                    }
    
            entries.append(entry)
        except:
            print("skipping " + row['MessageText'])
         
    if bad_lines:
        for l in bad_lines:
            print(l)
##        raise Exception('Bad slots found. See output above')
        
    return pd.DataFrame(entries, columns=['QueryId', 'RawQuery', 'Intent', 'Domain', 'SlotXML'])

def load_as_carina_and_embedding(filename):

    # create a local bert client
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
            sentenceEmbedding = bc.encode(query)

            # for debug 
            #print(f'sentenceEmbedding: {sentenceEmbedding.shape}')



            entry = {
                    'QueryId': 0,
                    'RawQuery': row['MessageText'],
                    'Intent': intent,
                    'Domain': domain,
                    'SlotXML': annotation,
                    'SentenceEmbedding':  sentenceEmbedding
                    }
    
            entries.append(entry)
        except:
            print("skipping " + row['MessageText'])
         
    if bad_lines:
        for l in bad_lines:
            print(l)
##        raise Exception('Bad slots found. See output above')
        
    return pd.DataFrame(entries, columns=['QueryId', 'RawQuery', 'Intent', 'Domain', 'SlotXML', 'SentenceEmbedding'])

if __name__ == "__main__":

    slotFiles = glob.glob("*.tsv");
    for slotFile in slotFiles:
        # only data
        #df = load_as_carina(slotFile)

        df = load_as_carina_and_embedding(slotFile)

        # generate csv without embedding
        df.to_csv('.\\' + slotFile.replace('.tsv', '-carina.tsv'), sep='\t', index=None)    



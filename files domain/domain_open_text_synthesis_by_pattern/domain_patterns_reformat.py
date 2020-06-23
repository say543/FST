import glob;
import codecs;
import random;
import os
from shutil import copyfile


import string




outputFile = 'pattern.txt'
normal_tokens_files = 'normal_token.txt'

# deduplication and sort
#outputs = [];
#outputsWithSource = [];

outputs = set()
outputsWithSource = set()


tags = ['sharetarget_name', 'file_name', 'file_type', 'sharetarget_type', 'order_ref', 'to_contact_name', 'date', 'files_keyword',
            'file_action', 'file_action_context', 'file_keyword', 'file_folder', 'meeting_starttime', 'contact_name', 'file_recency','file_boost', 'data_source']



PREVIOUSTURNDOMAIN = "PreviousTurnDomain"
PREVIOUSTURNINTENT = "PreviousTurnIntent"
TASKFRAMESTATUS = "TaskFrameStatus"
TASKFRAMEENTITYSTATES = "TaskFrameEntityStates"
TASKFRAMEGUID = "TaskFrameGUID"
SPEECHPEOPLEDISAMBIGUATIONGRAMMARMATCHES = "SpeechPeopleDisambiguationGrammarMatches"
CONVERSATIONALCONTEXT = "ConversationalContext"

FILLINEMPTYCOLUMN = set([
    PREVIOUSTURNDOMAIN,
    PREVIOUSTURNINTENT,
    TASKFRAMESTATUS,
    TASKFRAMEENTITYSTATES,
    TASKFRAMEGUID,
    SPEECHPEOPLEDISAMBIGUATIONGRAMMARMATCHES,
    CONVERSATIONALCONTEXT
    ])

TARGETNUMCOLUMNS = 2


normal_tokens = set()

patternWithHolder = []

isHeadColumn = True
headColumnList =[] 
with codecs.open('patterns_no_placeholder.txt', 'r', 'utf-8') as fin:

    print('####verify ####')
    print(fin.name)
    print('####verify ####')
    
    for line in fin:


        #skip headcolumn and check if valid
        if (isHeadColumn):
            line = line.strip();
            if not line:
                continue;
                   
            headColumnList = line.split('\t');

            # for debug
            #print(file)
            #print(headColumnList)
                    
            if len(headColumnList) < TARGETNUMCOLUMNS:
                    print("error header for file: " + str(len(headColumnList)));
                    
            isHeadColumn = False
            continue
        
        line = line.strip()
        if not line:
            continue

        #using  splace as default
        array = line.split('\t')

        pattern = array[0]

        ##using  splace as default

        # for deubg
        #print(pattern)

        ##using  splace as default
        tokens = pattern.split()

        isvalid = True

        query =""
        xml = ""
        for token in tokens:

            # for deubg
            #print(token)
            if (token.startswith('<') and token.endswith('>')):
                tag = token[1:len(token)-1]
                if tag not in tags:
                    print("error:" + line)
                    isvalid = False
                else:
                    if (tag.lower() == 'file_boost'):
                        query = query + " " + ("".join(tag.split('_'))).upper() + 'HOLDER'
                        xml = xml +  " " + ("".join(tag.split('_'))).upper() + 'HOLDER'                            
                    else:
                        query = query + " " + ("".join(tag.split('_'))).upper() + 'HOLDER'
                        xml = xml + " " +token.lower() + " " + ("".join(tag.split('_'))).upper() + 'HOLDER' + " " + '</' + tag.lower() + '>'
            else:
                # fir debug
                #print(pattern)
                normal_tokens.add(token)

                query = query +" " + token
                xml = xml +" " + token



        if isvalid:
            # here intent is default file_search not sure since for domain
            #outputs.append('0'+'\t'+query.strip()+ '\t'+ 'file_search' + '\t' +'files' +'\t' +xml.strip())
            outputs.add('0'+'\t'+query.strip()+ '\t'+ 'file_search' + '\t' +'files' +'\t' +xml.strip())



with codecs.open(normal_tokens_files, 'w', 'utf-8') as fout:
    for item in sorted(normal_tokens):
        fout.write(item + '\r\n');


#outputs = ['\t'.join(['id', 'query', 'intent', 'domain', 'QueryXml'])] + outputs;

with codecs.open(outputFile, 'w', 'utf-8') as fout:
    for item in sorted(outputs):
        fout.write(item + '\r\n');


'''
# remove unnecessary columns since they are empty
#outputs = ['\t'.join(['id', 'query', 'intent', 'domain', 'QueryXml', 'id', '0'])] + outputs;
#outputsWithSource = ['\t'.join(['id', 'query', 'intent', 'domain', 'QueryXml', 'id', '0', 'source'])] + outputsWithSource;

outputs = ['\t'.join(['id', 'query', 'intent', 'domain', 'QueryXml'])] + outputs;
outputsWithSource = ['\t'.join(['id', 'query', 'intent', 'domain', 'QueryXml', 'source'])] + outputsWithSource;







with codecs.open(outputFile, 'w', 'utf-8') as fout:
    for item in outputs:
        fout.write(item + '\r\n');

with codecs.open(outputFileWithSource, 'w', 'utf-8') as fout:
    for item in outputsWithSource:
        fout.write(item + '\r\n');

# replace directly
# if do not want , comment this
with codecs.open(outputTrainingFolderFile, 'w', 'utf-8') as fout:
    for item in outputs:
        fout.write(item + '\r\n');



# STCA test folder
with codecs.open(outputSTCATrainingFolderFile, 'w', 'utf-8') as fout:
    for item in outputs:
        fout.write(item + '\r\n');

'''

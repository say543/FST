import glob;
import codecs;
import random;
import os
import math
import re
import sys


from shutil import copyfile


# random seed
# and inner loop random seed change
rand_seed_parameter_initialization = 0.1
rand_seed_offset = 0.01


outputFile = 'files_measurement_training.tsv'

outputFileUnique = 'files_measurement_unique_training.tsv'
# replace directly
#outputTrainingFolderFile = '..\\files_slot_training.tsv'
# for STCA test
#outputSTCATrainingFolderFile = '..\\sharemodeltest\\files_slot_training.tsv'

outputFileWithSource = "files_measurement_training_with_source.tsv"

#outputFileKeyWordAndFileNameLexiconFilfe = 'filekeyword_filename_lexicon.txt'

# leave it but actually it is not being used
dsatTraining = "dsat_training.tsv"

files = glob.glob("*.tsv");
outputs = [];
outputsSet = set([]);

outputsWithSource = [];



# default file type is not found
# using list instead of set to access by index
#defaultFileTypeifMissed =set([
#    'file',
#    'files',
#    'document',
#    'documents'
#    ])

defaultFileTypeifMissed =[
    'file',
    'files',
    'document',
    'documents'
]



fileKeyWordAndFileNameCandidateSet = set()
# no need to collect file type. if needed, open it in the fturue
#fileTypeCandidateSet = set()

############################################
# copy file from synthetic data
############################################

############################################
# copy file from data folder directly
############################################




#initial rand seed
rand_seed_parameter = rand_seed_parameter_initialization


ConversationIdPrefix = 'files_domain_measurement'
ConversationId = 0;

for file in files:

    if file == outputFile or file == outputFileWithSource or file == outputFileUnique:
        continue;

    # skip dsat training at first
    #if file == dsatTraining:
    #    continue;
    
    print("collecting: " + file);
    with codecs.open(file, 'r', 'utf-8') as fin:
        for line in fin:
            line = line.strip();
            if not line:
                continue;
            array = line.split('\t');
            if len(array) < 5:
                print("error:" + line);


            

            #ConversationId,	MessageId, MessageTimestamp, MessageFrom, 'MessageText', 'JudgedIntent', 'JudgedDomain', 'JudgedConstraints', 'ConversationContext', 'MetaData', 'Frequency'  'Cat'

            #print(ConversationIdPrefix +"_"+str(ConversationId)+'\t' +"0"+'\t'+'\t'+'user'+'\t'+ array[1] +'\t' + array[2] + '\t' + array[3] + '\t' + array[4] + '\t'+"{}"+'\t'+'[]'+'\t'+'1'+'\t'+"")    
            outputs.append(ConversationIdPrefix +"_"+str(ConversationId)+'\t' +"0"+'\t'+'\t'+'user'+'\t'+ array[1] +'\t' + array[2] + '\t' + array[3].upper() + '\t' + array[4] + '\t'+"{}"+'\t'+'[]'+'\t'+'1'+'\t'+"");


            outputsSet.add(ConversationIdPrefix +"_"+str(ConversationId)+'\t' +"0"+'\t'+'\t'+'user'+'\t'+ array[1] +'\t' + array[2] + '\t' + array[3].upper() + '\t' + array[4] + '\t'+"{}"+'\t'+'[]'+'\t'+'1'+'\t'+"")

            outputsWithSource.append(ConversationIdPrefix +"_"+str(ConversationId)+'\t' +"0"+'\t'+'\t'+'user'+'\t'+ array[1] +'\t' + array[2] + '\t' + array[3].upper() + '\t' + array[4] + '\t'+"{}"+'\t'+'[]'+'\t'+'1'+'\t'+""+'\t'+ file);

            ConversationId+=1

# remove unnecessary columns since they are empty
#outputs = ['\t'.join(['id', 'query', 'intent', 'domain', 'QueryXml', 'id', '0'])] + outputs;
#outputsWithSource = ['\t'.join(['id', 'query', 'intent', 'domain', 'QueryXml', 'id', '0', 'source'])] + outputsWithSource;

# output soted order for easy check
outputs = ['\t'.join(['ConversationId','MessageId','MessageTimestamp', 'MessageFrom', 'MessageText', 'JudgedIntent', 'JudgedDomain', 'JudgedConstraints', 'ConversationContext', 'MetaData','Frequency','Cat'])] + sorted(outputs);
outputsWithSource = ['\t'.join(['ConversationId', 'MessageId','MessageTimestamp', 'MessageFrom', 'MessageText','JudgedIntent', 'JudgedDomain', 'JudgedConstraints', 'ConversationContext', 'MetaData', 'Frequency','Cat','source'])] + sorted(outputsWithSource);




with codecs.open(outputFile, 'w', 'utf-8') as fout:
    for item in outputs:
        fout.write(item + '\r\n');


print("dedup size = ")
print(len(outputsSet))
with codecs.open(outputFileUnique, 'w', 'utf-8') as fout:
    fout.write('\t'.join(['ConversationId','MessageId','MessageTimestamp','MessageFrom', 'MessageText', 'JudgedIntent', 'JudgedDomain', 'JudgedConstraints', 'ConversationContext', 'MetaData','Frequency','Cat']) + '\r\n');
    for item in sorted(outputsSet):
        fout.write(item + '\r\n');


print(len(outputsSet))
with codecs.open(outputFileWithSource, 'w', 'utf-8') as fout:
    for item in outputsWithSource:
        fout.write(item + '\r\n');


#with codecs.open(outputFileKeyWordAndFileNameLexiconFilfe, 'w', 'utf-8') as fout:
#    # sort it to easy check
#    for item in sorted(fileKeyWordAndFileNameCandidateSet):
#        fout.write(item + '\r\n');


# replace directly
# if do not want , comment this
#with codecs.open(outputTrainingFolderFile, 'w', 'utf-8') as fout:
#    for item in outputs:
#        fout.write(item + '\r\n');



# STCA test folder
#with codecs.open(outputSTCATrainingFolderFile, 'w', 'utf-8') as fout:
#    for item in outputs:
#        fout.write(item + '\r\n');

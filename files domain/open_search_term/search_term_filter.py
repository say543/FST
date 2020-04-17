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


outputFile = 'filekeyword_filetype_training.tsv'

outputFileUnique = 'search_term_less_than_three_unique_training.tsv'
# replace directly
#outputTrainingFolderFile = '..\\files_slot_training.tsv'
# for STCA test
#outputSTCATrainingFolderFile = '..\\sharemodeltest\\files_slot_training.tsv'

outputFileWithSource = "filekeyword_filetype_training_with_source.tsv"

outputFileKeyWordAndFileNameLexiconFilfe = 'filekeyword_filename_lexicon.txt'

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

            # for debug
            #print(line)
            
            if not line:
                continue;


            array = line.split(' ');

            # if longer than 3 ignore
            if len(array) > 3 :
                #print("ignore:" + line);
                continue
            
            #unique for dedup
            print(line)
            outputsSet.add(line);

            


print("dedup size = ")
print(len(outputsSet))
with codecs.open(outputFileUnique, 'w', 'utf-8') as fout:
    #fout.write('\t'.join(['id', 'query', 'intent', 'domain', 'QueryXml']) + '\r\n');
    for item in sorted(outputsSet):
        fout.write(item + '\r\n');


#print(len(outputsSet))
#with codecs.open(outputFileWithSource, 'w', 'utf-8') as fout:
#    for item in outputsWithSource:
#        fout.write(item + '\r\n');


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

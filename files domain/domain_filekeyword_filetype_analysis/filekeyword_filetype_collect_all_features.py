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

outputFileUnique = 'filekeyword_filetype_unique_training.tsv'
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

    if file == outputFile or file == outputFileWithSource:
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

            # skip last column for indicating which dataset
            #if file == dsatTraining:
            #    line ='\t'.join(array[0:len(array)-1])
            #outputs.append(line);
            #outputsWithSource.append(line+'\t'+ file);


            # update seed for each query
            rand_seed_parameter=rand_seed_parameter+rand_seed_offset;
            random.seed(rand_seed_parameter);


            #add new here
            slot = array[4]
            # for extra target slot you want
            xmlpairs = re.findall("(<.*?>.*?<\/.*?>)", slot)


            # heuristic solution
            # for one query
            # only extra a pair of file_keyword/ file_name and file_type


            # default using null
            fileKeyWordAndFileNameXml = None
            fileTypeXml = None
            fileKeyWordAndFileName = None
            fileType = None
            
            for xmlpair in xmlpairs:

                # extra type and value for xml tag
                xmlTypeEndInd = xmlpair.find(">")

                xmlType = xmlpair[1:xmlTypeEndInd]

                xmlValue = xmlpair.replace("<"+xmlType+">", "")
                xmlValue = xmlValue.replace("</"+xmlType+">", "")
                xmlValue = xmlValue.strip()


                
                if xmlType.lower() == "file_keyword" or xmlType.lower() == "file_name":
                    fileKeyWordAndFileNameCandidateSet.add(xmlValue)
                    if fileKeyWordAndFileNameXml is None:
                        fileKeyWordAndFileNameXml = xmlpair
                        fileKeyWordAndFileName = xmlValue
                if xmlType.lower() == "file_type":
                    if fileTypeXml is None:
                        fileTypeXml = xmlpair
                        fileType = xmlValue

            # document if filekeyword or filename exist
            if fileKeyWordAndFileNameXml is not None:

                # generate filetype if missed
                # for those cases they are do not have xml_tag
                if fileTypeXml is None:
                    indexInRange = random.randint(0, len(defaultFileTypeifMissed)-1)
                    #fileTypeXml =  "<file_type>" +  defaultFileTypeifMissed[indexInRange] + "</file_type>"
                    fileTypeXml =  defaultFileTypeifMissed[indexInRange]
                    fileType = defaultFileTypeifMissed[indexInRange]
                
                
            # for dsatTraining
            # replace file name with the last column
            if file == dsatTraining:
                newline ='\t'.join(array[0:len(array)-1])
                newfile = array[len(array)-1]
                outputs.append(newline);
                outputsWithSource.append(newline+'\t'+ newfile);
            else:

                if fileKeyWordAndFileNameXml is not None and fileTypeXml is not None:
                    #'id', 'query', 'intent', 'domain', 'QueryXml'
                    outputs.append(array[0]+'\t'+ fileKeyWordAndFileName + " " + fileType +'\t' + "file_search" + '\t' + array[3] + '\t' + fileKeyWordAndFileNameXml + " " + fileTypeXml);

                    #unique for dedup
                    outputsSet.add(array[0]+'\t'+ fileKeyWordAndFileName + " " + fileType +'\t' + "file_search" + '\t' + array[3] + '\t' + fileKeyWordAndFileNameXml + " " + fileTypeXml);
                    
                    #'id', 'query', 'intent', 'domain', 'QueryXml','source'
                    outputsWithSource.append(array[0]+'\t'+ fileKeyWordAndFileName + " " + fileType +'\t' + "file_search" + '\t' + array[3] + '\t' + fileKeyWordAndFileNameXml + " " + fileTypeXml+'\t'+file);
                
                #outputs.append(line);
                #outputsWithSource.append(line+'\t'+ file);
            



# remove unnecessary columns since they are empty
#outputs = ['\t'.join(['id', 'query', 'intent', 'domain', 'QueryXml', 'id', '0'])] + outputs;
#outputsWithSource = ['\t'.join(['id', 'query', 'intent', 'domain', 'QueryXml', 'id', '0', 'source'])] + outputsWithSource;

# output soted order for easy check
outputs = ['\t'.join(['id', 'query', 'intent', 'domain', 'QueryXml'])] + sorted(outputs);
outputsWithSource = ['\t'.join(['id', 'query', 'intent', 'domain', 'QueryXml', 'source'])] + sorted(outputsWithSource);




with codecs.open(outputFile, 'w', 'utf-8') as fout:
    for item in outputs:
        fout.write(item + '\r\n');


print("dedup size = ")
print(len(outputsSet))
with codecs.open(outputFileUnique, 'w', 'utf-8') as fout:
    fout.write('\t'.join(['id', 'query', 'intent', 'domain', 'QueryXml']) + '\r\n');
    for item in sorted(outputsSet):
        fout.write(item + '\r\n');


print(len(outputsSet))
with codecs.open(outputFileWithSource, 'w', 'utf-8') as fout:
    for item in outputsWithSource:
        fout.write(item + '\r\n');


with codecs.open(outputFileKeyWordAndFileNameLexiconFilfe, 'w', 'utf-8') as fout:
    # sort it to easy check
    for item in sorted(fileKeyWordAndFileNameCandidateSet):
        fout.write(item + '\r\n');


# replace directly
# if do not want , comment this
#with codecs.open(outputTrainingFolderFile, 'w', 'utf-8') as fout:
#    for item in outputs:
#        fout.write(item + '\r\n');



# STCA test folder
#with codecs.open(outputSTCATrainingFolderFile, 'w', 'utf-8') as fout:
#    for item in outputs:
#        fout.write(item + '\r\n');

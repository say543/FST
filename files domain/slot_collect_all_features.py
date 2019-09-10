import glob;
import codecs;
import random;
import os
from shutil import copyfile





outputFile = 'files_slot_training.tsv'
# replace directly
outputTrainingFolderFile = '..\\files_slot_training.tsv'
outputFileWithSource = "files_slot_training_with_source.tsv"
files = glob.glob("*.tsv");
outputs = [];
outputsWithSource = [];


# copy file from data folder directly
# update more in the future
copyfile("..\\files_mystuff_after_filtering.tsv" , "files_mystuff_after_filtering.tsv")
copyfile("..\\teams_slot_training_after_filtering.tsv" , "teams_slot_training_after_filtering.tsv")


for file in files:

    if file == outputFile or file == outputFileWithSource:
        continue;
    
    print("collecting: " + file);
    with codecs.open(file, 'r', 'utf-8') as fin:
        for line in fin:
            line = line.strip();
            if not line:
                continue;
            array = line.split('\t');
            if len(array) < 5:
                print("error:" + line);
            outputs.append(line);
            outputsWithSource.append(line+'\t'+ file);

print('shuffling');
random.seed(0.1);
random.shuffle(outputs);

outputs = ['\t'.join(['id', 'query', 'intent', 'domain', 'QueryXml', 'id', '0'])] + outputs;
outputsWithSource = ['\t'.join(['id', 'query', 'intent', 'domain', 'QueryXml', 'id', '0', 'source'])] + outputsWithSource;




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

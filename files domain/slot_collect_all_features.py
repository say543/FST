import glob;
import codecs;
import random;
import os


outputFile = 'files_slot_training.tsv'
files = glob.glob("*.tsv");
outputs = [];

for file in files:

    if file == outputFile:
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

print('shuffling');
random.seed(0.1);
random.shuffle(outputs);

outputs = ['\t'.join(['id', 'query', 'intent', 'domain', 'QueryXml', 'id', '0'])] + outputs;




with codecs.open(outputFile, 'w', 'utf-8') as fout:
    for item in outputs:
        fout.write(item + '\r\n');

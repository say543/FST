import glob;
import codecs;
import random;

files = glob.glob("*.tsv");
outputs = [];

for file in files:
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

with codecs.open('Teams_Slot_Training.tsv', 'w', 'utf-8') as fout:
    for item in outputs:
        fout.write(item + '\r\n');

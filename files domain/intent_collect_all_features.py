import glob;
import codecs;
import random;

outputFile = 'files_intent_training.tsv'
outputFileWithSource = "files_intent_training_with_source.tsv"
files = glob.glob("*.tsv");
outputs = [];
outputsWithSource = [];

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
            if len(array) < 4:
                print("error:" + line);
            outputs.append(line);
            outputsWithSource.append(line+'\t'+ file);

print('shuffling');
random.seed(0.1);
random.shuffle(outputs);

#TurnNumber	PreviousTurnIntent	query	intent
outputs = ['\t'.join(['TurnNumber', 'PreviousTurnIntent', 'query', 'intent'])] + outputs;
outputsWithSource = ['\t'.join(['TurnNumber', 'PreviousTurnIntent', 'query', 'intent', 'source'])] + outputsWithSource;


with codecs.open(outputFile, 'w', 'utf-8') as fout:
    for item in outputs:
        fout.write(item + '\r\n');


with codecs.open(outputFileWithSource, 'w', 'utf-8') as fout:
    for item in outputsWithSource:
        fout.write(item + '\r\n');

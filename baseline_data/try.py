import codecs;
outputSet = [];

with codecs.open('file_title.tsv', 'r', 'utf-8') as fin:
    for line in fin:
        line = line.strip();
        if not line:
            continue;
        outputSet.append(line);

outSet = outputSet[::2];

with codecs.open('file_title_seperated.tsv', 'w', 'utf-8') as fout:
    for item in outSet:
        fout.write(item + '\r\n');

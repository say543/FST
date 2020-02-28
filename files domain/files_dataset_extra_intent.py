import codecs;
import random;

# add hyper paramter if unbalanced
hyper_parameter = 200

# add repeated times
repated_time = 30


fileDomainRelatedIntent = ['file_search', 'file_open', 'file_share', 'file_download', 'file_other']

OutputSet = [];

with codecs.open('files_dataset.tsv', 'r', 'utf-8') as fin:
    for line in fin:
        line = line.strip();
        if not line:
            continue;
        linestrs = line.split("\t")
        if len(linestrs) < 5:
            continue;
        #for i in range(0,repated_time):
        #    OutputSet.append(linestrs[0]+"\t\t"+linestrs[1]+"\t"+linestrs[2]);
        # add
        # "PreviousTurnDomain"
        # "PreviousTurnIntent"
        # as
        # 'TurnNumber', PREVIOUSTURNINTENT, 'query', 'intent',PREVIOUSTURNDOMAIN])
        # append empty at first
        for i in range(0,repated_time):
            OutputSet.append(linestrs[0]+"\t\t"+linestrs[1]+"\t"+linestrs[2]+"\t");
            

"""
# comment shuffle in the first place
#random.shuffle(OutputSet);
"""

with codecs.open('files_dataset_intent.tsv', 'w', 'utf-8') as fout:
    for item in OutputSet:
        fout.write(item + '\r\n');

        

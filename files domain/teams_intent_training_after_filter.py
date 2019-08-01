import codecs;
import random;

# add hyper paramter if unbalanced
hyper_parameter = 200

# 07132019 add file_navigate
fileDomainRelatedIntent = ['file_search', 'file_open', 'file_share', 'file_download', 'file_other']
# no need to add file nvatigate for intent training
# only slot is needed
#fileDomainRelatedIntent = ['file_search', 'file_open', 'file_share', 'file_download', 'file_other', 'file_navigate']

OutputSet = [];

with codecs.open('teams_intent_training.tsv', 'r', 'utf-8') as fin:
    for line in fin:
        line = line.strip();
        if not line:
            continue;
        linestrs = line.split("\t");
        if len(linestrs) < 4:
            continue;
        
        if linestrs[3] in fileDomainRelatedIntent:

            # if file_navigate
            
            OutputSet.append(line);

"""
# comment shuffle in the first place
#random.shuffle(OutputSet);
"""

with codecs.open('teams_intent_training_after_filtering.tsv', 'w', 'utf-8') as fout:
    for item in OutputSet:
        fout.write(item + '\r\n');

        

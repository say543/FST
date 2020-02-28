import codecs;
import random;
import re;


import math
import re
import sys

# add hyper paramter if unbalanced
hyper_parameter = 200



commonDomainRelatedIntent = ['cancel','confirm','reject','select_none']

#inmeetingDomainToFileDomain = {
#    "inmeeting" : "files"
#}




OutputSet = [];
with codecs.open('InMeeting_Intent_Training_01292020v1.tsv', 'r', 'utf-8') as fin:
    for line in fin:
        line = line.strip();
        if not line:
            continue;
        linestrs = line.split("\t")
        if len(linestrs) < 4:
            continue;

        # make sure it is in common intent

        # add
        # "PreviousTurnDomain"
        # "PreviousTurnIntent"
        # as
        # 'TurnNumber', PREVIOUSTURNINTENT, 'query', 'intent',PREVIOUSTURNDOMAIN])
        # append empty at first
        if linestrs[3] in commonDomainRelatedIntent:        
            OutputSet.append(linestrs[0]+"\t\t"+linestrs[2]+"\t"+linestrs[3]+"\t");


# commna shuffle at first
#random.shuffle(OutputSet);

with codecs.open('inmeeting_intent_training_after_extract.tsv', 'w', 'utf-8') as fout:
    fout.write('\t'.join(['TurnNumber', 'PreviousTurnIntent', 'query', 'intent','PreviousTurnDomain'])+'\r\n');
    for item in OutputSet:
        fout.write(item + '\r\n');


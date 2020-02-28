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
with codecs.open('calendar.train.intentsslots.generated.tsv', 'r', 'utf-8') as fin:
    for line in fin:
        line = line.strip();
        if not line:
            continue;
        linestrs = line.split("\t")
        if len(linestrs) < 4:
            continue;


        # for debug 
        #print(len(linestrs))
        #print(line)

        # make sure it is in common intent
        #if linestrs[3] in commonDomainRelatedIntent:        
        #    OutputSet.append(linestrs[0]+"\t\t"+linestrs[2]+"\t"+linestrs[3]);

        # add
        # "PreviousTurnDomain"
        # "PreviousTurnIntent"
        # as
        # 'TurnNumber', PREVIOUSTURNINTENT, 'query', 'intent',PREVIOUSTURNDOMAIN])
        # since calendar some queries might miss two columns
        # append empty at first
        if linestrs[3] in commonDomainRelatedIntent:        
            #OutputSet.append(linestrs[0]+"\t"+linestrs[7]+"\t"+linestrs[2]+"\t"+linestrs[3]+"\t"+linestrs[6]);
            OutputSet.append(linestrs[0]+"\t\t"+linestrs[2]+"\t"+linestrs[3]+"\t");
        

# commna shuffle at first
#random.shuffle(OutputSet);

with codecs.open('calendar_intent_training_after_extract.tsv', 'w', 'utf-8') as fout:
    fout.write('\t'.join(['TurnNumber', 'PreviousTurnIntent', 'query', 'intent','PreviousTurnDomain'])+'\r\n');
    for item in OutputSet:
        fout.write(item + '\r\n');


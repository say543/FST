import codecs;
import random;

# add hyper paramter if unbalanced
hyper_parameter = 200



commonDomainRelatedIntent = ['cancel','confirm','reject','select_none']

rewriteIntent = "file_other" 

OutputSet = [];
with codecs.open('files_other_training.tsv', 'r', 'utf-8') as fin:
    for line in fin:
        line = line.strip();
        if not line:
            continue;
        linestrs = line.split("\t")
        if len(linestrs) < 3:
            continue;

        #skip common domain intent
        if linestrs[3] not in commonDomainRelatedIntent:
            OutputSet.append(linestrs[0]+"\t\t"+linestrs[2]+"\t"+rewriteIntent);        

# commna shuffle at first
#random.shuffle(OutputSet);

with codecs.open('files_other_training_after_rewrite.tsv', 'w', 'utf-8') as fout:
    for item in OutputSet:
        fout.write(item + '\r\n');
    


        

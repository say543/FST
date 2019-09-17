import codecs;
import random;

# add hyper paramter if unbalanced
hyper_parameter = 200

# add repeated times
repated_time = 10


fileDomainRelatedIntent = ['file_search', 'file_open', 'file_share', 'file_download', 'file_other']

Output = [];

with codecs.open('files_mystuff_after_filtering.tsv', 'r', 'utf-8') as fin:
    for line in fin:
        line = line.strip();
        if not line:
            continue;
        linestrs = line.split("\t")
        if len(linestrs) < 5:
            continue;
        

        if linestrs[2] in fileDomainRelatedIntent:

            # following the guideline to ove queries to file_navigate intent
            verbs = set(["go to ",
                        "Go to ",
                        "navigate to ",
                        "Navigate to ",
                        ])


            for verb in verbs:
                if linestrs[1].find(verb) != -1:
                    linestrs[2] = "file_navigate"
                    break
        
        for i in range(0,repated_time):
            Output.append(linestrs[0]+"\t\t"+linestrs[1]+"\t"+linestrs[2]);

"""
# comment shuffle in the first place
#random.shuffle(OutputSet);
"""

with codecs.open('files_mystuff_after_filtering_intent.tsv', 'w', 'utf-8') as fout:
    for item in Output:
        fout.write(item + '\r\n');

        

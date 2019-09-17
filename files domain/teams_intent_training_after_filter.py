import codecs;
import random;

# add hyper paramter if unbalanced
hyper_parameter = 200

# 07132019 add file_navigate
#fileDomainRelatedIntent = ['file_search', 'file_open', 'file_share', 'file_download', 'file_other']
# no need to add file nvatigate for intent training
# only slot is needed
# add teamspace_search but filter with different rules 
#fileDomainRelatedIntent = ['file_search', 'file_open', 'file_share', 'file_download', 'file_other', 'file_navigate']
fileDomainRelatedIntent = ['file_search', 'file_open', 'file_share', 'file_download', 'file_other', 'file_navigate', 'teamspace_search']


##############################
# intent level candidate
##############################
teamspaceSearchCandidateSet = set()


Output = [];

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

            # document teamspace_search
            # for further analysis
            # skip it at first
            if linestrs[3] == "teamspace_search":
                # having problem or tab
                # then skip
                if linestrs[2].find("program") != -1 or \
                   linestrs[2].find("tab") != -1 or \
                   linestrs[2].find("channel") != -1 or \
                   linestrs[2].find("team") != -1 or \
                   linestrs[2].find("teams") != -1 or \
                   linestrs[2].find("conversation") != -1 or \
                   linestrs[2].find("chat") != -1:
                    teamspaceSearchCandidateSet.add(line)
                    continue
                linestrs[3] = "file_search"

            # following the guideline to move queries to file_search intent
            verbs = set(["search ",
                        "Search ",
                        "find ",
                        "show ",
                        "Show ",
                        "Open ",
                        "open ",
                        "Find ",
                        "find ",
                        "display ",
                        "Display ",
                        "locate ",
                        "Locate ",
                        ])

            if linestrs[3] == "file_navigate":

                for verb in verbs:
                    if linestrs[2].find(verb) != -1:
                        linestrs[3] = "file_search"
                        break

            # TurnNumber	PreviousTurnIntent	query	intent
            Output.append(linestrs[0]+"\t"+linestrs[1]+"\t"+linestrs[2]+"\t"+linestrs[3]);

"""
# comment shuffle in the first place
#random.shuffle(OutputSet);
"""


#######################
# intent level output
#######################
with codecs.open('teams_intent_training_after_filtering.tsv', 'w', 'utf-8') as fout:
    for item in Output:
        fout.write(item + '\r\n');


        
with codecs.open('teams_intent_training_after_filtering_teamspace_search.tsv', 'w', 'utf-8') as fout:
    for item in teamspaceSearchCandidateSet:
        fout.write(item + '\r\n');

        

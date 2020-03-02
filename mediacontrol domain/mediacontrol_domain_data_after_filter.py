import codecs;
import random;
import re;
import glob;
import os;


import math
import re
import sys

from collections import defaultdict;

# add hyper paramter if unbalanced
hyper_parameter = 200



fileDomainRelatedIntent = ['file_search', 'file_open', 'file_share', 'file_download', 'file_other', 'file_navigate', "teamspace_search"]

teamsDomainToFileDomain = {
    # carina need small but bellevue needs big
    "teams" : "files",
    "TEAMS" : "FILES"
}




#OutputSlotEvaluation = [];

#OutputIntentEvaluation = [];


#OutputSTCAIntentEvaluation = [];


domainListDictionary = defaultdict(list);


# validaitng opened dataset
inputFile = "mediacontrol_domain_train.tsv"


#with codecs.open('Teams-golden.tsv', 'r', 'utf-8') as fin:
with codecs.open(inputFile, 'r', 'utf-8') as fin:
    
    for line in fin:
        line = line.strip();
        if not line:
            continue;
        linestrs = line.split("\t");

        # make sure it at least has
        # Query	ExternalFeature	Weight	Intent	Domain	Slot
        if len(linestrs) < 4:
            continue;


        domain = linestrs[3]

        if domain not in domainListDictionary:
            domainListDictionary[domain] = []
        domainListDictionary[domain].append(line)
        
        


        # TurnNumber / PreviousTurnIntent / query /intent

        # id / message / intent / domain / constraint
        # for training purpose's format
            
        ##OutputSlotEvaluation.append("0"+"\t"+linestrs[4]+"\t"+linestrs[7]+"\t" +linestrs[6].lower()+"\t"+linestrs[8]);

        # TurnNumber / PreviousTurnIntent / query /intent
        # for training purpose's format
        ##OutputIntentEvaluation.append("0"+"\t"+""+"\t"+linestrs[4]+"\t" +linestrs[7]);

        #UUID\tQuery\tIntent\tDomain\tSlot\r\n
        #OutputSTCAIntentEvaluation.append("0"+"\t"+linestrs[4]+"\t" +linestrs[6]+"\t"+linestrs[5].lower()+"\t"+linestrs[7])

        #id\tquery\tintent\tdomain\tQueryXml\r\n"
        ##OutputSTCAIntentEvaluation.append("0"+"\t"+linestrs[4]+"\t" +linestrs[7]+"\t"+linestrs[6].lower()+"\t"+linestrs[8])



        

"""
# comment shuffle in the first place
#random.shuffle(OutputSet);
"""

# outout to different bucket for analysis
for domain, rows in domainListDictionary.items():
    with codecs.open((inputFile.split("."))[0] +'_'+domain+'.tsv', 'w', 'utf-8') as fout:
        # header
        fout.write("TurnNumber\PreviousTurnDomain\tquery\domain\r\n")
        for row in rows:
            fout.write(row + '\r\n');
        


# for judge trainer format
#with codecs.open('teams_golden_after_filtering.tsv', 'w', 'utf-8') as fout:
#
#    # if outout originla format
#    fout.write("ConversationId\tMessageId\tMessageTimestamp\tMessageFrom\tMessageText\tJudgedDomain\tJudgedIntent\tJudgedConstraints\tMetaData\tConversationContext\tFrequency\tImplicitConstraints\r\n")
#    for item in Output:
#        fout.write(item + '\r\n');

# for CMF slot evaluation format
'''
with codecs.open((inputFile.split("."))[0] +'slot_evaluation.tsv', 'w', 'utf-8') as fout:

    # if output for traing
    fout.write("id\tquery\tintent\tdomain\tQueryXml\r\n")
    for item in OutputSlotEvaluation:
        fout.write(item + '\r\n');
'''

# for STCA evaluation
'''
with codecs.open("sharemodeltest\\"+(inputFile.split("."))[0] +'slot_evaluation.tsv', 'w', 'utf-8') as fout:

    # if output for traing
    fout.write("id\tquery\tintent\tdomain\tQueryXml\r\n")
    for item in OutputSlotEvaluation:
        fout.write(item + '\r\n');
'''

# for CMF intent evaluation format
'''
with codecs.open((inputFile.split("."))[0] +'intent_evaluation.tsv', 'w', 'utf-8') as fout:

    # if output for traing
    fout.write("TurnNumber\tPreviousTurnIntent\tquery\tintent\r\n")
    for item in OutputIntentEvaluation:
        fout.write(item + '\r\n');
'''

# for STCAevaluation
'''
with codecs.open("sharemodeltest\\"+(inputFile.split("."))[0] +'intent_evaluation.tsv', 'w', 'utf-8') as fout:

    print("sharemodeltest\\"+(inputFile.split("."))[0] +'intent_evaluation.tsv')

    # if output for traing
    #fout.write("UUID\tQuery\tIntent\tDomain\tSlot\r\n")
    fout.write("id\tquery\tintent\tdomain\tQueryXml\r\n")
    for item in OutputSTCAIntentEvaluation:
        fout.write(item + '\r\n');
'''

#######################
# intent level output
#######################
'''
with codecs.open('teams_slot_training_after_filtering_teamspace_search.tsv', 'w', 'utf-8') as fout:
    for item in teamspaceSearchCandidateSet:
        fout.write(item + '\r\n');
'''


#######################
# slot level output
#######################

'''
with codecs.open('teams_slot_training_after_filtering_file_keyword.tsv', 'w', 'utf-8') as fout:
    for item in fileKeywordCandidateSet:
        fout.write(item + '\r\n');

with codecs.open('teams_slot_training_after_filtering_file_name.tsv', 'w', 'utf-8') as fout:
    for item in fileNameCandidateSet:
        fout.write(item + '\r\n');

with codecs.open('teams_slot_training_after_filtering_meeting_starttime.tsv', 'w', 'utf-8') as fout:
    for item in meetingStarttimeCandidateSet:
        fout.write(item + '\r\n');

with codecs.open('teams_slot_training_after_filtering_file_type.tsv', 'w', 'utf-8') as fout:
    for item in fileTypeCandidateSet:
        fout.write(item + '\r\n');

# this is to deduplication
with codecs.open('teams_slot_training_after_filtering_file_recency.tsv', 'w', 'utf-8') as fout:
    for item in fileRecencyCandidateSet:
        fout.write(item + '\r\n');

with codecs.open('teams_slot_training_after_filtering_sharetarget_type.tsv', 'w', 'utf-8') as fout:
    for item in sharetargetTypeCandidateSet:
        fout.write(item + '\r\n');

with codecs.open('teams_slot_training_after_filtering_sharetarget_name.tsv', 'w', 'utf-8') as fout:
    for item in sharetargetNameCandidateSet:
        fout.write(item + '\r\n');

with codecs.open('teams_slot_training_after_filtering_contact_name.tsv', 'w', 'utf-8') as fout:
    for item in contactNameCandidateSet:
        fout.write(item + '\r\n');

with codecs.open('teams_slot_training_after_filtering_file_action.tsv', 'w', 'utf-8') as fout:
    for item in fileActionCandidateSet:
        fout.write(item + '\r\n');
        
with codecs.open('teams_slot_training_after_filtering_order_ref.tsv', 'w', 'utf-8') as fout:
    for item in orderRefCandidateSet:
        fout.write(item + '\r\n');
'''

#######################
# query replacement revert
#######################

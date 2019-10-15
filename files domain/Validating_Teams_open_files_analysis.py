import codecs;
import random;
import re;
import glob;
import os;


import math
import re
import sys

# add hyper paramter if unbalanced
hyper_parameter = 200



fileDomainRelatedIntent = ['file_search', 'file_open', 'file_share', 'file_download', 'file_other', 'file_navigate', "teamspace_search"]

teamsDomainToFileDomain = {
    # carina need small but bellevue needs big
    "teams" : "files",
    "TEAMS" : "FILES"
}


openFileExtention ={
"pptx",
"ppts",
"ppt",
"deck",
"decks",
"presentation",
"presentations",
"powerpoint",
"powerpoints",
"power point",
"slide",
"slides",
"doc",
"docx",
"docs",
"Doc",
"Docx",
"Docs",
"spec",
"excel",
"excels",
"xls",
"xlsx",
"spreadsheet",
"spreadsheets",
"workbook",
"worksheet",
"csv",
"tsv",
"note",
"notes",
"onenote",
"onenotes",
"OneNote",
"notebook",
"notebooks",
"pdf",
"pdfs",
"PDF",
"jpg",
"jpeg",
"gif",
"png",
"image",
"msg",
"ics",
"vcs",
"vsdx",
"vssx",
"vstx",
"vsdm",
"vssm",
"vstm",
"vsd",
"vdw",
"vss",
"vst",
"mpp",
"mpt",
"word",
"words",
"document",
"documents",
"file",
"files"
}



OutputOpenEvaluation = [];

OutputOpenExtentionEvaluation = [];



inputFile = "Validating_Teams_20190716-20190831_16k.tsv"

with codecs.open(inputFile, 'r', 'utf-8') as fin:
    
    for line in fin:
        line = line.strip();
        if not line:
            continue;
        linestrs = line.split("\t");

        # make sure it at least has
        # Query	ExternalFeature	Weight	Intent	Domain	Slot
        if len(linestrs) < 11:
            continue;

        if linestrs[4].lower().find("open") != -1:

            OutputOpenEvaluation.append(linestrs[4]+"\t"+linestrs[5]+"\t"+linestrs[10]);
            
            found = False
            for key in sorted (openFileExtention):
                #print(key)
                if linestrs[4].lower().find(key.lower()) != -1:
                    #print(linestrs[4])
                    found = True
                    break

            if found:
                #print(linestrs[4])
                OutputOpenExtentionEvaluation.append(linestrs[4]+"\t"+linestrs[5]+"\t"+linestrs[10]);
            
                    

        

        #if linestrs[5] != 'FILES':
        #    continue

        # id / message / intent / domain / constraint
        # for training purpose's format
            
        #OutputSlotEvaluation.append("0"+"\t"+linestrs[4]+"\t"+linestrs[6]+"\t" +linestrs[5].lower()+"\t"+linestrs[7]);

        # TurnNumber / PreviousTurnIntent / query /intent
        # for training purpose's format
        #OutputIntentEvaluation.append("0"+"\t"+""+"\t"+linestrs[4]+"\t" +linestrs[6]);

        #UUID\tQuery\tIntent\tDomain\tSlot\r\n
        #OutputSTCAIntentEvaluation.append("0"+"\t"+linestrs[4]+"\t" +linestrs[6]+"\t"+linestrs[5].lower()+"\t"+linestrs[7])

"""
# comment shuffle in the first place
#random.shuffle(OutputSet);
"""


# for STCAevaluation 
with codecs.open((inputFile.split("."))[0] +'open_evaluation.tsv', 'w', 'utf-8') as fout:

    # if output for traing
    fout.write("Query\tDomain\tFrequency\r\n")
    for item in OutputOpenEvaluation:
        fout.write(item + '\r\n');

with codecs.open((inputFile.split("."))[0] +'open_extention_evaluation.tsv', 'w', 'utf-8') as fout:

    # if output for traing
    fout.write("Query\tDomain\tFrequency\r\n")
    for item in OutputOpenExtentionEvaluation:
        fout.write(item + '\r\n');



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

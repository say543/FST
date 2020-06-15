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


domain = "files"


##############################
# canonical merge
##############################
# dictionary of dictionary
# with sub dictionary for each canonical value
cano={
    }

#
#canofiles = glob.glob(r"*.txt");
#canofiles = ["file_recency.txt", "file_type.txt", "order_ref.txt", "position_ref.txt", "sharetarget_type.txt", "file_folder.txt","file_keyword.txt","sharetarget_name.txt"];
# add contact_name , to_contact_name
canofiles = ["file_recency.txt", "file_type.txt", "order_ref.txt", "position_ref.txt", "sharetarget_type.txt", "file_folder.txt","file_keyword.txt","sharetarget_name.txt","contact_name.txt","to_contact_name.txt"];


outputFile = 'files.canonical.value.tsv'
for file in canofiles:

    # skip output file
    if file == outputFile:
        continue;
    
    filestr = os.path.basename(file)
    filestr = filestr.split('.');
    #print(filestr[0])

    key =filestr[0]
    cano[key] = {}

    

    print("collecting: " + file + " for:" + key );
    with codecs.open(file, 'r', 'utf-8') as fin:
        for line in fin:
            line = line.strip();
            if not line:
                continue;
            array = line.split('\t');
            if len(array) < 2:
                print("error:" + line);

            cano[key][array[0]] = array[1]
    



with codecs.open(outputFile, 'w', 'utf-8') as fout:

    # if outout originla format
    fout.write("Domain\tSlotName\tSlotValue\tCanonicalValue\r\n")
    for slot in cano:
        for slotValue in cano[slot]:
            #print(slot)
            #print(slotValue)
            fout.write(domain + '\t' + slot + '\t' + slotValue + '\t' + cano[slot][slotValue] + '\r\n');

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

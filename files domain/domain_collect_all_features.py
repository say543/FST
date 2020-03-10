import glob;
import codecs;
import random;
import os
from shutil import copyfile





outputFile = 'files_domain_training.tsv'
# replace directly
outputTrainingFolderFile = '..\\files_domain_training.tsv'
# for STCA test
outputSTCATrainingFolderFile = '..\\sharemodeltest\\files_domain_training.tsv'

outputFileWithSource = "files_domain_training_with_source.tsv"
#dsatTraining = "dsat_training.tsv"

files = glob.glob("*.tsv");
outputs = [];
outputsWithSource = [];


PREVIOUSTURNDOMAIN = "PreviousTurnDomain"
PREVIOUSTURNINTENT = "PreviousTurnIntent"
TASKFRAMESTATUS = "TaskFrameStatus"
TASKFRAMEENTITYSTATES = "TaskFrameEntityStates"
TASKFRAMEGUID = "TaskFrameGUID"
SPEECHPEOPLEDISAMBIGUATIONGRAMMARMATCHES = "SpeechPeopleDisambiguationGrammarMatches"
CONVERSATIONALCONTEXT = "ConversationalContext"

FILLINEMPTYCOLUMN = set([
    PREVIOUSTURNDOMAIN,
    PREVIOUSTURNINTENT,
    TASKFRAMESTATUS,
    TASKFRAMEENTITYSTATES,
    TASKFRAMEGUID,
    SPEECHPEOPLEDISAMBIGUATIONGRAMMARMATCHES,
    CONVERSATIONALCONTEXT
    ])

TARGETNUMCOLUMNS = 10



############################################
# copy file from synthetic data
############################################



# cancel each one if you do n
#copyfile("..\\Open_Text_Synthesis\\file_type_and_contact_name\\data_synthesised_contact_name_file_type.tsv" , "data_synthesised_contact_name_file_type.tsv")
#copyfile("..\\Open_Text_Synthesis\\file_keyword_and_file_type\\data_synthesised_file_keyword_file_type.tsv" , "data_synthesised_file_keyword_file_type.tsv")
#copyfile("..\\Open_Text_Synthesis\\file_keyword_and_to_contact_name\\data_synthesised_to_contact_name_file_keyword.tsv" , "data_synthesised_to_contact_name_file_keyword.tsv")
#copyfile("..\\Open_Text_Synthesis\\file_keyword_sharetarget_type_sharetarget_name_file_type_to_contact_name\\data_synthesised_file_keyword_sharetarget_type_sharetarget_name_file_type_to_contact_name.tsv",\
#         "data_synthesised_file_keyword_sharetarget_type_sharetarget_name_file_type_to_contact_name.tsv")
#copyfile("..\\Open_Text_Synthesis\\file_type_and_to_contact_name\\data_synthesised_to_contact_name_file_type.tsv" , "data_synthesised_to_contact_name_file_type.tsv")

#copyfile("..\\Open_Text_Synthesis\\file_keyword_and_contact_name\\data_synthesised_contact_name_file_keyword.tsv" , "data_synthesised_contact_name_file_keyword.tsv")

#copyfile("..\\Open_Text_Synthesis\\file_keyword_file_type_file_action_contact_name\\data_synthesised_file_keyword_file_type_file_action_contact_name_file_name_verb.tsv" , "data_synthesised_file_keyword_file_type_file_action_contact_name_file_name_verb.tsv")

#copyfile("..\\Open_Text_Synthesis\\file_type_and_contact_name_to_contact_name\\data_synthesised_contact_name_file_type_to_contact_name.tsv" , "data_synthesised_contact_name_file_type_to_contact_name.tsv")

#copyfile("..\\Open_Text_Synthesis\\file_keyword_file_type_to_contact_name\\data_synthesised_file_keyword_file_type_to_contact_name.tsv" , "data_synthesised_file_keyword_file_type_to_contact_name.tsv")


# add for send <to_contact_name>  <contact_name> <file_keyword> <file_type> synthesis data
#copyfile("..\\Open_Text_Synthesis\\file_keyword_file_type_to_contact_name\\data_synthesised_file_keyword_file_type_to_contact_name.tsv" , "data_synthesised_file_keyword_file_type_to_contact_name.tsv")

# comment this since date should be part of file_keyword


# reopen it but merge date into filekeyword into a single slots
# open it for domain training
copyfile("..\\Open_Text_Synthesis\\file_keyword_file_type_file_action_contact_name_date\\domain_data_synthesised_file_keyword_file_type_file_action_contact_name_file_name_verb_date.tsv" , "domain_data_synthesised_file_keyword_file_type_file_action_contact_name_file_name_verb_date.tsv")




############################################
# copy file from data folder directly
############################################
#copyfile("..\\files_dataset.tsv" , "files_dataset.tsv")
#copyfile("..\\files_mystuff_after_filtering.tsv" , "files_mystuff_after_filtering.tsv")
#copyfile("..\\teams_slot_training_after_filtering.tsv" , "teams_slot_training_after_filtering.tsv")
#copyfile("..\\dsat_training.tsv" , "dsat_training.tsv")


copyfile("..\\mediacontrol_domain\\mediacontrol_domain_train_after_filter.tsv" , "mediacontrol_domain_train_after_filter.tsv")
copyfile("..\\domain_dsat_training.tsv" , "domain_dsat_training.tsv")

for file in files:

    if file == outputFile or file == outputFileWithSource:
        continue;

    # skip dsat training at first
    #if file == dsatTraining:
    #    continue;

    
    
    print("collecting: " + file);
    with codecs.open(file, 'r', 'utf-8') as fin:

        if file == 'mediacontrol_domain_train_after_filter.tsv' or file == 'domain_dsat_training.tsv':

            isHeadColumn = True
            headColumnList =[] 
            for line in fin:
                #skip headcolumn and check if valid
                if (isHeadColumn):
                    line = line.strip();
                    if not line:
                        continue;
                    headColumnList = line.split('\t');
                    if len(headColumnList) < TARGETNUMCOLUMNS:
                        print("error header for file: " + synlist);
                    
                    isHeadColumn = False
                    continue

            
                line = line.strip();
                if not line:
                    continue;
                array = line.split('\t');

                # append space is smaller than target length
                lineWithFill = line
                if len(array) < TARGETNUMCOLUMNS:

                    lineWithFill =""
                    for index in range(0,TARGETNUMCOLUMNS):

                        if index >= len(array):
                            # debug
                            #print(index)
                            if headColumnList[index] in FILLINEMPTYCOLUMN:
                                lineWithFill = lineWithFill
                            else:
                                print("error:" + line);
                        else:
                            lineWithFill = lineWithFill+array[index]

                        if index < TARGETNUMCOLUMNS-1:
                            lineWithFill+="\t";
                    

                        
                            #lineWithFill = "\t"+lineWithFill+array[index]


                    #lineWithFill = lineWithFill.rstrip()
                outputs.append(lineWithFill)
                outputsWithSource.append(lineWithFill+'\t'+ file);
        


                # for dsatTraining
                # replace file name with the last column
                # open this if in the future has domain dsat training
                #if file == dsatTraining:
                #    newline ='\t'.join(array[0:len(array)-1])
                #    newfile = array[len(array)-1]
                #    outputs.append(newline);
                #    outputsWithSource.append(newline+'\t'+ newfile);
            #else:
            #    outputs.append(line);
            #    outputsWithSource.append(line+'\t'+ file);

        else:
            # for other synthesis data
            # no heder provided so far
            # change in the future
            headColumns = 'TurnNumber\tPreviousTurnIntent\tquery\tdomain\tPreviousTurnDomain\tTaskFrameStatus\tTaskFrameEntityStates\tTaskFrameGUID\tSpeechPeopleDisambiguationGrammarMatches\tConversationalContext'
            headColumnList = headColumns.split('\t');
            for line in fin:

                line = line.strip();
                if not line:
                    continue;
                array = line.split('\t');
                if len(array) < 5:
                    print("error:" + line);

                linestrs = line.split('\t')

                #extra partial columns and fill in
                #'TurnNumber', PREVIOUSTURNINTENT, 'query', 'domain',PREVIOUSTURNDOMAIN, TASKFRAMESTATUS, TASKFRAMEENTITYSTATES, TASKFRAMEGUID, SPEECHPEOPLEDISAMBIGUATIONGRAMMARMATCHES, CONVERSATIONALCONTEXT]
                lineWithFill = linestrs[0]+"\t\t"+ linestrs[1]+'\t'+linestrs[3]


                # for debug
                #print(lineWithFill)

                array = lineWithFill.split('\t');
                if len(array) < TARGETNUMCOLUMNS:

                    lineWithFill =""
                    for index in range(0,TARGETNUMCOLUMNS):

                        if index >= len(array):
                            # debug
                            #print(index)
                            if headColumnList[index] in FILLINEMPTYCOLUMN:
                                lineWithFill = lineWithFill
                            else:
                                print("error:" + line);
                        else:
                            lineWithFill = lineWithFill+array[index]

                        if index < TARGETNUMCOLUMNS-1:
                            lineWithFill+="\t";
                    

                        
                            #lineWithFill = "\t"+lineWithFill+array[index]


                    #lineWithFill = lineWithFill.rstrip()
                outputs.append(lineWithFill)
                outputsWithSource.append(lineWithFill+'\t'+ file);

print('shuffling');
random.seed(0.1);
random.shuffle(outputs);


# remove unnecessary columns since they are empty
#outputs = ['\t'.join(['id', 'query', 'intent', 'domain', 'QueryXml', 'id', '0'])] + outputs;
#outputsWithSource = ['\t'.join(['id', 'query', 'intent', 'domain', 'QueryXml', 'id', '0', 'source'])] + outputsWithSource;

outputs = ['\t'.join(['TurnNumber', PREVIOUSTURNINTENT, 'query', 'domain',PREVIOUSTURNDOMAIN, TASKFRAMESTATUS, TASKFRAMEENTITYSTATES, TASKFRAMEGUID, SPEECHPEOPLEDISAMBIGUATIONGRAMMARMATCHES, CONVERSATIONALCONTEXT])] + outputs;
outputsWithSource = ['\t'.join(['TurnNumber', PREVIOUSTURNINTENT, 'query', 'domain',PREVIOUSTURNDOMAIN, TASKFRAMESTATUS, TASKFRAMEENTITYSTATES, TASKFRAMEGUID, SPEECHPEOPLEDISAMBIGUATIONGRAMMARMATCHES, CONVERSATIONALCONTEXT])] + outputsWithSource;



with codecs.open(outputFile, 'w', 'utf-8') as fout:
    for item in outputs:
        fout.write(item + '\r\n');

with codecs.open(outputFileWithSource, 'w', 'utf-8') as fout:
    for item in outputsWithSource:
        fout.write(item + '\r\n');

# replace directly
# if do not want , comment this
with codecs.open(outputTrainingFolderFile, 'w', 'utf-8') as fout:
    for item in outputs:
        fout.write(item + '\r\n');



# STCA test folder
with codecs.open(outputSTCATrainingFolderFile, 'w', 'utf-8') as fout:
    for item in outputs:
        fout.write(item + '\r\n');

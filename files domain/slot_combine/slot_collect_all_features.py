import glob;
import codecs;
import random;
import os
from shutil import copyfile





outputFile = 'files_slot_training.tsv'
# replace directly
outputTrainingFolderFile = '..\\files_slot_training.tsv'
# for STCA test
outputSTCATrainingFolderFile = '..\\sharemodeltest\\files_slot_training.tsv'

outputFileWithSource = "files_slot_training_with_source.tsv"
dsatTraining = "dsat_training.tsv"

files = glob.glob("*.tsv");
outputs = [];
outputsWithSource = [];

############################################
# copy file from synthetic data
############################################
# cancel each one if you do n
copyfile("..\\Open_Text_Synthesis\\file_type_and_contact_name\\data_synthesised_contact_name_file_type.tsv" , "data_synthesised_contact_name_file_type.tsv")
copyfile("..\\Open_Text_Synthesis\\file_keyword_and_file_type\\data_synthesised_file_keyword_file_type.tsv" , "data_synthesised_file_keyword_file_type.tsv")
copyfile("..\\Open_Text_Synthesis\\file_keyword_and_to_contact_name\\data_synthesised_to_contact_name_file_keyword.tsv" , "data_synthesised_to_contact_name_file_keyword.tsv")
copyfile("..\\Open_Text_Synthesis\\file_keyword_sharetarget_type_sharetarget_name_file_type_to_contact_name\\data_synthesised_file_keyword_sharetarget_type_sharetarget_name_file_type_to_contact_name.tsv",\
         "data_synthesised_file_keyword_sharetarget_type_sharetarget_name_file_type_to_contact_name.tsv")
copyfile("..\\Open_Text_Synthesis\\file_type_and_to_contact_name\\data_synthesised_to_contact_name_file_type.tsv" , "data_synthesised_to_contact_name_file_type.tsv")

copyfile("..\\Open_Text_Synthesis\\file_keyword_and_contact_name\\data_synthesised_contact_name_file_keyword.tsv" , "data_synthesised_contact_name_file_keyword.tsv")

copyfile("..\\Open_Text_Synthesis\\file_keyword_file_type_file_action_contact_name\\data_synthesised_file_keyword_file_type_file_action_contact_name_file_name_verb.tsv" , "data_synthesised_file_keyword_file_type_file_action_contact_name_file_name_verb.tsv")

copyfile("..\\Open_Text_Synthesis\\file_type_and_contact_name_to_contact_name\\data_synthesised_contact_name_file_type_to_contact_name.tsv" , "data_synthesised_contact_name_file_type_to_contact_name.tsv")

copyfile("..\\Open_Text_Synthesis\\file_keyword_file_type_to_contact_name\\data_synthesised_file_keyword_file_type_to_contact_name.tsv" , "data_synthesised_file_keyword_file_type_to_contact_name.tsv")


# add for send <to_contact_name>  <contact_name> <file_keyword> <file_type> synthesis data
copyfile("..\\Open_Text_Synthesis\\file_keyword_file_type_to_contact_name\\data_synthesised_file_keyword_file_type_to_contact_name.tsv" , "data_synthesised_file_keyword_file_type_to_contact_name.tsv")

# comment this since date should be part of file_keyword


# reopen it but merge date into filekeyword into a single slots
copyfile("..\\Open_Text_Synthesis\\file_keyword_file_type_contact_name_to_contact_name\\data_synthesised_file_keyword_to_contact_name_file_type_contact_name.tsv" , "data_synthesised_file_keyword_to_contact_name_file_type_contact_name.tsv")



# file SERP file_type from contact name date
copyfile("..\\Open_Text_Synthesis\\file_type_contact_name_date_and_time\\data_synthesised_file_type_contact_name_date_time.tsv" , "data_synthesised_file_type_contact_name_date_time.tsv")



# attachment new features for intent models
# if features works well, turn it on as normal data
copyfile("..\\Open_Text_Synthesis\\contact_name_order_ref_and_file_recency\\data_synthesised_contact_name_order_ref_file_recency.tsv", "data_synthesised_contact_name_order_ref_file_recency.tsv")


# file_keyword_file_type_file_action_to_contact_name
copyfile("..\\Open_Text_Synthesis\\file_keyword_file_type_file_action_to_contact_name\\data_synthesised_file_keyword_file_type_file_action_to_contact_name.tsv" , "data_synthesised_file_keyword_file_type_file_action_to_contact_name.tsv")


# add file_keyword
#not too much help so comment it at first
#copyfile("..\\Open_Text_Synthesis\\file_keyword\\data_synthesised_file_keyword.tsv" , "data_synthesised_file_keyword.tsv")


# file_boost_file_action_file_keyword_file_recency_contact_name_to_contact_name
# cannot find so rename to domain
#copyfile("..\\Open_Text_Synthesis\\file_boost_file_action_file_keyword_file_recency_contact_name_to_contact_name\\data_synthesised_file_boost_file_action_file_keyword_file_recency_to_contact_name_contact_name.tsv" , "data_synthesised_file_boost_file_action_file_keyword_file_recency_to_contact_name_contact_name.tsv")
copyfile("..\\Open_Text_Synthesis\\file_boost_file_action_file_keyword_file_recency_contact_name_to_contact_name\\domain.tsv" , "data_synthesised_file_boost_file_action_file_keyword_file_recency_to_contact_name_contact_name.tsv")

############################################
# copy file from data folder directly
############################################
copyfile("..\\files_dataset.tsv" , "files_dataset.tsv")
copyfile("..\\files_mystuff_after_filtering.tsv" , "files_mystuff_after_filtering.tsv")
copyfile("..\\teams_slot_training_after_filtering.tsv" , "teams_slot_training_after_filtering.tsv")
copyfile("..\\dsat_training.tsv" , "dsat_training.tsv")

for file in files:

    if file == outputFile or file == outputFileWithSource:
        continue;

    # skip dsat training at first
    #if file == dsatTraining:
    #    continue;
    
    print("collecting: " + file);
    with codecs.open(file, 'r', 'utf-8') as fin:
        for line in fin:
            line = line.strip();
            if not line:
                continue;
            array = line.split('\t');
            if len(array) < 5:
                print("error:" + line);

            # skip last column for indicating which dataset
            #if file == dsatTraining:
            #    line ='\t'.join(array[0:len(array)-1])
            #outputs.append(line);
            #outputsWithSource.append(line+'\t'+ file);

            # for dsatTraining
            # replace file name with the last column
            if file == dsatTraining:
                newline ='\t'.join(array[0:len(array)-1])
                newfile = array[len(array)-1]
                outputs.append(newline);
                outputsWithSource.append(newline+'\t'+ newfile);
            else:
                outputs.append(line);
                outputsWithSource.append(line+'\t'+ file);
            

print('shuffling');
random.seed(0.1);
random.shuffle(outputs);


# remove unnecessary columns since they are empty
#outputs = ['\t'.join(['id', 'query', 'intent', 'domain', 'QueryXml', 'id', '0'])] + outputs;
#outputsWithSource = ['\t'.join(['id', 'query', 'intent', 'domain', 'QueryXml', 'id', '0', 'source'])] + outputsWithSource;

outputs = ['\t'.join(['id', 'query', 'intent', 'domain', 'QueryXml'])] + outputs;
outputsWithSource = ['\t'.join(['id', 'query', 'intent', 'domain', 'QueryXml', 'source'])] + outputsWithSource;




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

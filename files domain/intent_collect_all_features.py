import glob;
import codecs;
import random;

from shutil import copyfile

outputFile = 'files_intent_training.tsv'
outputTrainingFolderFile = '..\\files_intent_training.tsv'
outputFileWithSource = "files_intent_training_with_source.tsv"
files = glob.glob("*.tsv");
outputs = [];
outputsWithSource = [];



############################################
# copy file from synthetic data
# eg : copyfile("..\\Open_Text_Synthesis\\file_keyword_and_file_type\\data_synthesised_file_keyword_file_type.tsv" , "data_synthesised_file_keyword_file_type.tsv")
############################################
# cancel each one if you do n
#syn_hyperparameter = 2000
syn_hyperparameter = 500

synlists = set([
    "sharefile_synthesis.tsv",
    "share_synthesis.tsv",
    "send_synthesis.tsv",
    "downloadfire_synthesis.tsv",
    "showwith_synthesis.tsv",
    "openwith_synthesis.tsv",
    ])


for synlist in synlists:
    copyfile("..\\Intent_Synthesis\\"+synlist , synlist)
    temp =[]
    with codecs.open(synlist, 'r', 'utf-8') as fin:
        for line in fin:
            line = line.strip();
            if not line:
                continue;

            temp.append(line);

        print('shuffling:'+synlist);
        random.seed(0.1);
        random.shuffle(temp);
        
        for i in range(0,min(syn_hyperparameter, len(temp))):
            outputs.append(temp[i]);            
            outputsWithSource.append(temp[i]+'\t'+ synlist);
 

#copyfile("..\\Intent_Synthesis\\sharefile_synthesis.tsv" , "sharefile_synthesis.tsv")
#copyfile("..\\Intent_Synthesis\\share_synthesis.tsv" , "share_synthesis.tsv")
#copyfile("..\\Intent_Synthesis\\send_synthesis.tsv" , "send_synthesis.tsv")
#copyfile("..\\Intent_Synthesis\\downloadfire_synthesis.tsv" , "downloadfire_synthesis.tsv")

#copyfile("..\\Intent_Synthesis\\showwith_synthesis.tsv" , "showwith_synthesis.tsv")
#copyfile("..\\Intent_Synthesis\\openwith_synthesis.tsv" , "openwith_synthesis.tsv")

copyfile("..\\Open_Text_Synthesis\\file_keyword_and_to_contact_name\\intent_data_synthesised_to_contact_name_file_keyword.tsv" , "intent_data_synthesised_to_contact_name_file_keyword.tsv")
copyfile("..\\Open_Text_Synthesis\\file_keyword_file_type_file_action_contact_name\\intent_data_synthesised_file_keyword_file_type_file_action_contact_name_file_name_verb.tsv" , "intent_data_synthesised_file_keyword_file_type_file_action_contact_name_file_name_verb.tsv")

copyfile("..\\Open_Text_Synthesis\\file_type\\intent_data_synthesised_file_type.tsv", "intent_data_synthesised_file_type.tsv")


# not too much help so ignore it
#copyfile("..\\Open_Text_Synthesis\\file_keyword_and_file_type\\intent_data_synthesised_file_keyword_file_type.tsv", "intent_data_synthesised_file_keyword_file_type.tsv")

############################################
# copy file from data folder directly
############################################

copyfile("..\\files_other_training_after_rewrite.tsv", "files_other_training_after_rewrite.tsv")
copyfile("..\\files_dataset_intent.tsv" , "files_dataset_intent.tsv")
#remove copy my stuff data since no improvement so no update
#copyfile("..\\files_mystuff_after_filtering_intent.tsv" , "files_mystuff_after_filtering_intent.tsv")


copyfile("..\\teams_intent_training_after_filtering.tsv" , "teams_intent_training_after_filtering.tsv")





############################################
# copy synthesis data
############################################

for file in files:
    if file == outputFile or file == outputFileWithSource:
        continue;

    # my stuff needs to have to make all as search default
    #if file == "files_mystuff_after_filtering_intent.tsv" or file == "files_mystuff_after_filtering_intent_backup.tsv":
    if file == "files_mystuff_after_filtering_intent_backup.tsv":
        continue

    # skip synthesis data
    #if file == "sharefile_synthesis.tsv" or file == "share_synthesis.tsv" or file == "send_synthesis.tsv" \
    #or file == "downloadfire_synthesis.tsv" or file == "showwith_synthesis.tsv" or file == "openwith_synthesis.tsv":
    if file in synlists:
        continue
    
    print("collecting: " + file);
    with codecs.open(file, 'r', 'utf-8') as fin:
        for line in fin:
            line = line.strip();
            if not line:
                continue;
            array = line.split('\t');
            if len(array) < 4:
                print("error:" + line);
            
            outputs.append(line);            
            outputsWithSource.append(line+'\t'+ file);

print('shuffling');
random.seed(0.1);
random.shuffle(outputs);

#TurnNumber	PreviousTurnIntent	query	intent
outputs = ['\t'.join(['TurnNumber', 'PreviousTurnIntent', 'query', 'intent'])] + outputs;
outputsWithSource = ['\t'.join(['TurnNumber', 'PreviousTurnIntent', 'query', 'intent', 'source'])] + outputsWithSource;


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

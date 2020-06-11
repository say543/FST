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
# eg : copyfile("..\\Open_Text_Synthesis\\file_keyword_and_file_type\\data_synthesised_file_keyword_file_type.tsv" , "data_synthesised_file_keyword_file_type.tsv")

############################################
# cancel each one if you do n
#syn_hyperparameter = 2000
syn_hyperparameter = 500




# replace this routine by fixing a dataset

synlists = set([
    # will be capped by syn_hyperparameter. so rerun it might create problem 
    "sharefile_synthesis.tsv",
    "share_synthesis.tsv",
    "send_synthesis.tsv",
    "downloadfire_synthesis.tsv",
    "showwith_synthesis.tsv",
    "openwith_synthesis.tsv",
    ])

'''
for synlist in synlists:
    copyfile("..\\Intent_Synthesis\\"+synlist , synlist)


    print("collecting: " + synlist);
    isHeadColumn = True
    headColumnList =[] 

    
    temp =[]
    with codecs.open(synlist, 'r', 'utf-8') as fin:
        for line in fin:




            
            #line = line.strip();
            #if not line:
            #    continue;

            #temp.append(line);


            

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
            temp.append(lineWithFill)
                    

        print('shuffling:'+synlist);
        random.seed(0.1);
        random.shuffle(temp);

        #for debug
        print(len(temp))
        
        for i in range(0,min(syn_hyperparameter, len(temp))):
            outputs.append(temp[i]);            
            outputsWithSource.append(temp[i]+'\t'+ synlist);
'''



syn_fix_file = 'six_synthesis_selected_fix.tsv'
print("collecting: " + syn_fix_file);
isHeadColumn = True
headColumnList =[]
temp =[]
with codecs.open(syn_fix_file, 'r', 'utf-8') as fin:
    for line in fin:
  
        #line = line.strip();
        #if not line:
        #    continue;

        #temp.append(line);


        

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
        temp.append(lineWithFill)
            
    #for debug
    #print(len(temp))
    
    for i in range(0, len(temp)):
        outputs.append(temp[i]);        
        outputsWithSource.append(temp[i]+'\t'+ syn_fix_file);



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


# genearte new data from
copyfile("..\\Open_Text_Synthesis\\file_type_and_contact_name_to_contact_name\\intent_data_synthesised_to_contact_name_contact_name_file_type.tsv", "intent_data_synthesised_to_contact_name_contact_name_file_type.tsv")


# attachment new features for intent models
# if features works well, turn it on as normal data
copyfile("..\\Open_Text_Synthesis\\contact_name_order_ref_and_file_recency\\intent_data_synthesised_contact_name_order_ref_file_recency.tsv", "intent_data_synthesised_contact_name_order_ref_file_recency.tsv")


############################################
# copy file from data folder directly
############################################

copyfile("..\\files_other_training_after_rewrite.tsv", "files_other_training_after_rewrite.tsv")
copyfile("..\\files_dataset_intent.tsv" , "files_dataset_intent.tsv")
#remove copy my stuff data since no improvement so no update
# add back to extra modified intent result but change source to only has 7000 in case any wrong edtied operation
#copyfile("..\\files_mystuff_after_filtering_intent.tsv" , "files_mystuff_after_filtering_intent.tsv")
#copyfile("..\\files_mystuff_after_filtering_intent_modify_intent.tsv" , "files_mystuff_after_filtering_intent.tsv")
copyfile("..\\files_mystuff_after_filtering_intent_modify_intent.tsv" , "files_mystuff_after_filtering_intent_modify_intent.tsv")


copyfile("..\\teams_intent_training_after_filtering.tsv" , "teams_intent_training_after_filtering.tsv")

# add common intent data from inmeeting and calendar
copyfile("..\\inmeeting_intent_training_after_extract.tsv" , "inmeeting_intent_training_after_extract.tsv")
copyfile("..\\calendar_intent_training_after_extract.tsv" , "calendar_intent_training_after_extract.tsv")


# add multiturn training DASt files
# new column PREVIOUSTURNDOMAIN for the last one (PreviousTurnIntent  has already existed)
copyfile("..\\intent_dsat_training.tsv" , "intent_dsat_training.tsv")


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

    # has one open () * docx from vocie skill, not looks very promising query
    if file =='Validating_Teams_20191101-20191130_10k.tsv':
        continue

    # skip synthesis data
    #if file == "sharefile_synthesis.tsv" or file == "share_synthesis.tsv" or file == "send_synthesis.tsv" \
    #or file == "downloadfire_synthesis.tsv" or file == "showwith_synthesis.tsv" or file == "openwith_synthesis.tsv":
    if file in synlists or file == syn_fix_file:
        continue
    
    print("collecting: " + file);
    isHeadColumn = True
    headColumnList =[] 
    with codecs.open(file, 'r', 'utf-8') as fin:
        for line in fin:

            #skip headcolumn and check if valid
            if (isHeadColumn):
                line = line.strip();
                if not line:
                    continue;
                headColumnList = line.split('\t');
                if len(headColumnList) < TARGETNUMCOLUMNS:
                    print("error header for file: " + file);
                    
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


                    #for debug
                    #if file == 'files_mystuff_after_filtering_intent.tsv' and line == '0		"Cortana, I would like to view Minerals Slides saved last week."	file_open	':
                    #    for index in range(0,len(array)):
                    #        print(array[index])

                    if index >= len(array):
                        # for deubg
                        #print(index)
                        #if file == 'files_mystuff_after_filtering_intent.tsv':
                        #    print (line)
                        if headColumnList[index] in FILLINEMPTYCOLUMN:
                            lineWithFill = lineWithFill
                        else:
                            print("error:" + line);
                    else:
                        lineWithFill = lineWithFill+array[index]

                    if index < TARGETNUMCOLUMNS-1:
                        lineWithFill+="\t";


                #lineWithFill = lineWithFill.rstrip()
                    
            # miss column PreviousTurnDomain
            # append empty as default
            #if PREVIOUSTURNDOMAIN not in array:
            #    line = '\t'.join([line, ""])

            # miss column PreviousTurnDomain
            # append empty as default
            #if  not in array:
            #    line = '\t'.join([line, ""])

                        
            #outputs.append(line);            
            #outputsWithSource.append(line+'\t'+ file);

            
            outputs.append(lineWithFill);            
            outputsWithSource.append(lineWithFill+'\t'+ file);


print('shuffling');
random.seed(0.1);
random.shuffle(outputs);

#TurnNumber	PreviousTurnIntent	query	intent	PreviousTurnDomain
#outputs = ['\t'.join(['TurnNumber', 'PreviousTurnIntent', 'query', 'intent'])] + outputs;
#outputsWithSource = ['\t'.join(['TurnNumber', 'PreviousTurnIntent', 'query', 'intent', 'source'])] + outputsWithSource;
#outputs = ['\t'.join(['TurnNumber', PREVIOUSTURNINTENT, 'query', 'intent',PREVIOUSTURNDOMAIN])] + outputs;
#outputsWithSource = ['\t'.join(['TurnNumber', PREVIOUSTURNINTENT, 'query', 'intent', PREVIOUSTURNDOMAIN,'source'])] + outputsWithSource;

outputs = ['\t'.join(['TurnNumber', PREVIOUSTURNINTENT, 'query', 'intent',PREVIOUSTURNDOMAIN, TASKFRAMESTATUS, TASKFRAMEENTITYSTATES, TASKFRAMEGUID, SPEECHPEOPLEDISAMBIGUATIONGRAMMARMATCHES, CONVERSATIONALCONTEXT])] + outputs;
outputsWithSource = ['\t'.join(['TurnNumber', PREVIOUSTURNINTENT, 'query', 'intent',PREVIOUSTURNDOMAIN, TASKFRAMESTATUS, TASKFRAMEENTITYSTATES, TASKFRAMEGUID, SPEECHPEOPLEDISAMBIGUATIONGRAMMARMATCHES, CONVERSATIONALCONTEXT])] + outputsWithSource;




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

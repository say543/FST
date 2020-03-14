import glob;
import codecs;
import random;
import os
from shutil import copyfile





outputFile = 'domain_dsat_generate.tsv'
outputFileWithSource = 'domain_dsat_generate_with_source.tsv'

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
SOURCE= 'Source'
SOURCE2= 'Source2'

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

HYPERPARAMETER = 5



############################################
# copy file from synthetic data
############################################




for file in files:

    if file == outputFile or file == outputFileWithSource:
        continue;

    # skip dsat training at first
    #if file == dsatTraining:
    #    continue;

    
    
    print("collecting: " + file);
    with codecs.open(file, 'r', 'utf-8') as fin:

        # any name is fine , remove it in the futre
        if file == 'someone_with_head.txt': 

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

                fileSource = array[len(array)-1]

                # append space is smaller than target length
                #remove the last one
                lineWithFill = ""
                if len(array) < TARGETNUMCOLUMNS:
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
                else:
                    # skip data source
                    for index in range(0,TARGETNUMCOLUMNS):
                        lineWithFill = lineWithFill+array[index]
                        if index < TARGETNUMCOLUMNS-1:
                            lineWithFill+="\t";
                    
                outputs.append(lineWithFill)
                outputsWithSource.append(lineWithFill+'\t'+ fileSource);
        


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
            isHeadColumn = True
            headColumnList =[] 
            for line in fin:

                #skip headcolumn and check if valid
                if (isHeadColumn):
                    line = line.strip();
                    if not line:
                        continue;
                    headColumnList = line.split('\t');
                    
                    isHeadColumn = False
                    continue
                

                line = line.strip();
                if not line:
                    continue;
                array = line.split('\t');
                if len(array) < 3:
                    print("error:" + line);

                linestrs = line.split('\t')

                #extra partial columns and fill in
                #'TurnNumber', PREVIOUSTURNINTENT, 'query', 'domain',PREVIOUSTURNDOMAIN, TASKFRAMESTATUS, TASKFRAMEENTITYSTATES, TASKFRAMEGUID, SPEECHPEOPLEDISAMBIGUATIONGRAMMARMATCHES, CONVERSATIONALCONTEXT]
                lineWithFill = '0\t\t'+linestrs[0]+'\t'+linestrs[1].lower()+'\t\t\t\t\t\t\t'+linestrs[2];


                # for debug
                #print(lineWithFill)

                for i in range(0,HYPERPARAMETER):
                    outputs.append(lineWithFill)
                #outputsWithSource.append(lineWithFill+'\t'+ file);

#print('shuffling');
#random.seed(0.1);
#random.shuffle(outputs);


# remove unnecessary columns since they are empty
#outputs = ['\t'.join(['id', 'query', 'intent', 'domain', 'QueryXml', 'id', '0'])] + outputs;
#outputsWithSource = ['\t'.join(['id', 'query', 'intent', 'domain', 'QueryXml', 'id', '0', 'source'])] + outputsWithSource;

outputs = ['\t'.join(['TurnNumber', PREVIOUSTURNINTENT, 'query', 'domain',PREVIOUSTURNDOMAIN, TASKFRAMESTATUS, TASKFRAMEENTITYSTATES, TASKFRAMEGUID, SPEECHPEOPLEDISAMBIGUATIONGRAMMARMATCHES, CONVERSATIONALCONTEXT, SOURCE])] + outputs;
outputsWithSource = ['\t'.join(['TurnNumber', PREVIOUSTURNINTENT, 'query', 'domain',PREVIOUSTURNDOMAIN, TASKFRAMESTATUS, TASKFRAMEENTITYSTATES, TASKFRAMEGUID, SPEECHPEOPLEDISAMBIGUATIONGRAMMARMATCHES, CONVERSATIONALCONTEXT, SOURCE, SOURCE2])] + outputsWithSource;



with codecs.open(outputFile, 'w', 'utf-8') as fout:
    for item in outputs:
        fout.write(item + '\r\n');

#with codecs.open(outputFileWithSource, 'w', 'utf-8') as fout:
#    for item in outputsWithSource:
#        fout.write(item + '\r\n');

# replace directly
# if do not want , comment this
#with codecs.open(outputTrainingFolderFile, 'w', 'utf-8') as fout:
#    for item in outputs:
#        fout.write(item + '\r\n');



# STCA test folder
#with codecs.open(outputSTCATrainingFolderFile, 'w', 'utf-8') as fout:
#    for item in outputs:
#        fout.write(item + '\r\n');

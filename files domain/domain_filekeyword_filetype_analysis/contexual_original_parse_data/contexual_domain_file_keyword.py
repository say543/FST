import codecs;
import random;
import re;
import glob;
import os;


import math
import re
import sys

import json

# add hyper paramter if unbalanced
hyper_parameter = 200



# random seed
# and inner loop random seed change
rand_seed_parameter_initialization = 0.1
rand_seed_offset = 0.01


fileDomainRelatedIntent = ['file_search', 'file_open', 'file_share', 'file_download', 'file_other', 'file_navigate', "teamspace_search"]

teamsDomainToFileDomain = {
    # carina need small but bellevue needs big
    "teams" : "files",
    "TEAMS" : "FILES"
}



defaultFileTypeifMissed =[
        'document',
	'documents',
	'file',
	'files',
	'powerpoints',
	'power point',
	'slide',
	'slides',
	'doc',
	'docx',
	'docs',
	'spec',
	'excel',
	'excels',
	'xls',
	'xlsx',
	'spreadsheet',
	'spreadsheets',
	'workbook',
	'worksheet',
	'csv',
	'tsv',
	'note',
	'notes',
	'onenote',
	'onenotes',
	'onenote',
	'notebook',
	'notebooks',
	'pdf',
	'pdfs',
	'pdf',
	'jpg',
	'jpeg',
	'gif',
	'png',
	'image',
	'msg',
	'ics',
	'vcs',
	'vsdx',
	'vssx',
	'vstx',
	'vsdm',
	'vssm',
	'vstm',
	'vsd',
	'vdw',
	'vss',
	'vst',
	'mpp',
	'mpt',
	'word',
	'words'
#    'file',
#    'files',
#    'document',
#    'documents'
]


# for o(1) loook up
defaultFileTypeifMissedSet = set(defaultFileType.lower() for defaultFileType in defaultFileTypeifMissed)


outputs = [];
outputsSet = set([]);

outputsWithSource = [];

outputFileKeyWordAndFileNameUserContext = set([]);




fileKeyWordAndFileNameCandidateSet = set()


OutputSlotEvaluation = [];

OutputIntentEvaluation = [];


OutputSTCAIntentEvaluation = [];



# leave it but actually it is not being used
dsatTraining = "dsat_training.tsv"


outputFile = 'contexual_filekeyword_training.tsv'

outputFileUnique = 'contexual_filekeyword_unique_training.tsv'
# replace directly
#outputTrainingFolderFile = '..\\files_slot_training.tsv'
# for STCA test
#outputSTCATrainingFolderFile = '..\\sharemodeltest\\files_slot_training.tsv'

outputFileWithSource = "contexual_filekeyword_training_with_source.tsv"


'''
outputFileKeyWordAndFileNameLexiconFilfe = 'contexual_filekeyword_lexicon.txt'
outputFileKeyWordAndFileNameUserContextFiles = 'user_filekeyword_filename.txt'
'''

# validaitng opened dataset
# old format work
inputFile = "files_file_keyword_positive.tsv"




#initial rand seed
rand_seed_parameter = rand_seed_parameter_initialization


ConversationIdPrefix = 'contexual_filekeyword'
ConversationId = 0;


#with codecs.open('Teams-golden.tsv', 'r', 'utf-8') as fin:
with codecs.open(inputFile, 'r', 'utf-8') as fin:
    
    for line in fin:
        line = line.strip();
        if not line:
            continue;
        array = line.split('\t');

        # make sure it at least has
        # Query	ExternalFeature	Weight	Intent	Domain	Slot
        if len(array) < 6:
            continue;

        if array[1] != 'FILES':
            continue

        # update seed for each query
        rand_seed_parameter=rand_seed_parameter+rand_seed_offset;
        random.seed(rand_seed_parameter);

        #add new here
        slot = array[3]
        # for extra target slot you want
        xmlpairs = re.findall("(<.*?>.*?<\/.*?>)", slot)


        # heuristic solution
        # for one query
        # only extra a pair of file_keyword/ file_name and file_type


        # default using null
        fileKeyWordAndFileNameXml = None
        fileKeyWordAndFileName = None
            
        for xmlpair in xmlpairs:

            # extra type and value for xml tag
            xmlTypeEndInd = xmlpair.find(">")

            xmlType = xmlpair[1:xmlTypeEndInd]

            xmlValue = xmlpair.replace("<"+xmlType+">", "")
            xmlValue = xmlValue.replace("</"+xmlType+">", "")
            xmlValue = xmlValue.strip()


                
            if xmlType.lower() == "file_keyword" or xmlType.lower() == "file_name":
                fileKeyWordAndFileNameCandidateSet.add(xmlValue)
                if fileKeyWordAndFileNameXml is None:
                    fileKeyWordAndFileNameXml = xmlpair
                    fileKeyWordAndFileName = xmlValue

                                  
        # for dsatTraining
        # replace file name with the last column
        if inputFile == dsatTraining:
            newline ='\t'.join(array[0:len(array)-1])
            newfile = array[len(array)-1]
            outputs.append(newline);
            outputsWithSource.append(newline+'\t'+ newfile);
        else:

            if fileKeyWordAndFileNameXml is not None:
                '''
                #ConversationId,	MessageId, 'MessageText', 'JudgedIntent', 'JudgedDomain', 'JudgedConstraints', 'ConversationContext', 'Cat'
                
                #outputs.append("0"+'\t'+ fileKeyWordAndFileName + " " + fileType +'\t' + "file_search" + '\t' + array[1] + '\t' + fileKeyWordAndFileNameXml + " " + fileTypeXml+ '\t'+array[4]+'\t'+array[5]);
                outputs.append(ConversationIdPrefix +"_"+str(ConversationId)+'\t' +"0"+'\t'+ fileKeyWordAndFileName + " " + fileType +'\t' + "file_search" + '\t' + array[1] + '\t' + fileKeyWordAndFileNameXml + " " + fileTypeXml+ '\t'+array[4]+'\t'+array[5]);

                #unique for dedup
                #outputsSet.add("0"+'\t'+ fileKeyWordAndFileName + " " + fileType +'\t' + "file_search" + '\t' + array[1] + '\t' + fileKeyWordAndFileNameXml + " " + fileTypeXml+ '\t'+array[4]+'\t'+array[5]);
                outputsSet.add(ConversationIdPrefix +"_"+str(ConversationId)+'\t' +"0"+'\t'+ fileKeyWordAndFileName + " " + fileType +'\t' + "file_search" + '\t' + array[1] + '\t' + fileKeyWordAndFileNameXml + " " + fileTypeXml+ '\t'+array[4]+'\t'+array[5]);
                    
                #'id', 'query', 'intent', 'domain', 'QueryXml','source'
                #outputsWithSource.append("0"+'\t'+ fileKeyWordAndFileName + " " + fileType +'\t' + "file_search" + '\t' + array[1] + '\t' + fileKeyWordAndFileNameXml + " " + fileTypeXml+ '\t'+array[4]+'\t'+array[5]+'\t'+inputFile);
                outputsWithSource.append(ConversationIdPrefix +"_"+str(ConversationId)+'\t' +"0"+'\t'+ fileKeyWordAndFileName + " " + fileType +'\t' + "file_search" + '\t' + array[1] + '\t' + fileKeyWordAndFileNameXml + " " + fileTypeXml+ '\t'+array[4]+'\t'+array[5]+'\t'+inputFile);
                '''

                # handle special characater
                fileKeyWordAndFileName = fileKeyWordAndFileName.replace('&','&amp;')
                fileKeyWordAndFileNameXml = fileKeyWordAndFileNameXml.replace('&','&amp;')
                array[4] = array[4].replace('&','&amp;')
                
                

                #ConversationId,	MessageId, MessageTimestamp, MessageFrom, 'MessageText', 'JudgedIntent', 'JudgedDomain', 'JudgedConstraints', 'ConversationContext', 'MetaData', 'Frequency'  'Cat'
                
                #outputs.append("0"+'\t'+ fileKeyWordAndFileName + " " + fileType +'\t' + "file_search" + '\t' + array[1] + '\t' + fileKeyWordAndFileNameXml + " " + fileTypeXml+ '\t'+array[4]+'\t'+array[5]);
                #outputs.append(ConversationIdPrefix +"_"+str(ConversationId)+'\t' +"0"+'\t'+'\t'+'user'+'\t'+ fileKeyWordAndFileName + " " + fileType +'\t' + "file_search" + '\t' + array[1] + '\t' + fileKeyWordAndFileNameXml + " " + fileTypeXml+ '\t'+array[4]+'\t'+'[]'+'\t'+'1'+'\t'+array[5]);
                outputs.append(ConversationIdPrefix +"_"+str(ConversationId)+'\t' +"0"+'\t'+'\t'+'user'+'\t'+ fileKeyWordAndFileName + '\t' + "file_search" + '\t' + array[1] + '\t' + fileKeyWordAndFileNameXml + '\t'+array[4]+'\t'+'[]'+'\t'+'1'+'\t'+array[5]);

                #unique for dedup
                #outputsSet.add("0"+'\t'+ fileKeyWordAndFileName + " " + fileType +'\t' + "file_search" + '\t' + array[1] + '\t' + fileKeyWordAndFileNameXml + " " + fileTypeXml+ '\t'+array[4]+'\t'+array[5]);
                #outputsSet.add(ConversationIdPrefix +"_"+str(ConversationId)+'\t' +"0"+'\t'+'\t'+'user'+'\t'+ fileKeyWordAndFileName + " " + fileType +'\t' + "file_search" + '\t' + array[1] + '\t' + fileKeyWordAndFileNameXml + " " + fileTypeXml+ '\t'+array[4]+'\t'+'[]'+'\t'+'1'+'\t'+array[5]);
                outputsSet.add(ConversationIdPrefix +"_"+str(ConversationId)+'\t' +"0"+'\t'+'\t'+'user'+'\t'+ fileKeyWordAndFileName + '\t' + "file_search" + '\t' + array[1] + '\t' + fileKeyWordAndFileNameXml + '\t'+array[4]+'\t'+'[]'+'\t'+'1'+'\t'+array[5]);
                    
                #'id', 'query', 'intent', 'domain', 'QueryXml','source'
                #outputsWithSource.append("0"+'\t'+ fileKeyWordAndFileName + " " + fileType +'\t' + "file_search" + '\t' + array[1] + '\t' + fileKeyWordAndFileNameXml + " " + fileTypeXml+ '\t'+array[4]+'\t'+array[5]+'\t'+inputFile);
                #outputsWithSource.append(ConversationIdPrefix +"_"+str(ConversationId)+'\t' +"0"+'\t'+'\t'+'user'+'\t'+ fileKeyWordAndFileName + " " + fileType +'\t' + "file_search" + '\t' + array[1] + '\t' + fileKeyWordAndFileNameXml + " " + fileTypeXml+ '\t'+array[4]+'\t'+'[]'+'\t'+'1'+'\t'+array[5]+'\t'+inputFile);
                outputsWithSource.append(ConversationIdPrefix +"_"+str(ConversationId)+'\t' +"0"+'\t'+'\t'+'user'+'\t'+ fileKeyWordAndFileName + '\t' + "file_search" + '\t' + array[1] + '\t' + fileKeyWordAndFileNameXml + '\t'+array[4]+'\t'+'[]'+'\t'+'1'+'\t'+array[5]+'\t'+inputFile);





                ConversationId= ConversationId+1;




                # extra from json

                # to prevent from inside content apostrophy

                fileKeyWordAndFileNameXmlPostProcessing = array[4]

                #{'
                #':
                #['
                #',
                #, '
                #']
                # typical example
                #{'UserFileNames': ['Running Experiments', 'OneNote_DeletedPages', 'Running Experiments   
                
                fileKeyWordAndFileNameXmlPostProcessing = fileKeyWordAndFileNameXmlPostProcessing.replace("{'","{\"")
                fileKeyWordAndFileNameXmlPostProcessing = fileKeyWordAndFileNameXmlPostProcessing.replace("':","\":")
                fileKeyWordAndFileNameXmlPostProcessing = fileKeyWordAndFileNameXmlPostProcessing.replace("['","[\"")
                fileKeyWordAndFileNameXmlPostProcessing = fileKeyWordAndFileNameXmlPostProcessing.replace("',","\",")
                fileKeyWordAndFileNameXmlPostProcessing = fileKeyWordAndFileNameXmlPostProcessing.replace(", '",", \"")
                fileKeyWordAndFileNameXmlPostProcessing = fileKeyWordAndFileNameXmlPostProcessing.replace("']","\"]")



                # for deubg
                #print(fileKeyWordAndFileNameXmlPostProcessing)
       
                fileKeyWordAndFileNameXmlJson = json.loads(fileKeyWordAndFileNameXmlPostProcessing)

                #print(fileKeyWordAndFileNameXmlJson['UserFileNames'])

                #print(type(fileKeyWordAndFileNameXmlJson['UserFileNames']))

                for element in fileKeyWordAndFileNameXmlJson['UserFileNames']:
                    outputFileKeyWordAndFileNameUserContext.add(element)            
                
            #outputs.append(line);
            #outputsWithSource.append(line+'\t'+ file);
            


        
        '''
        # id / message / intent / domain / constraint / ConversationContext / cat
        # for training purpose's format
            
        OutputSlotEvaluation.append("0"+"\t"+linestrs[0]+"\t"+linestrs[2]+"\t" +linestrs[1].lower()+"\t"+linestrs[3]+"\t"+linestrs[4]+"\t"+linestrs[5]);

        # TurnNumber / PreviousTurnIntent / query /intent /ConversationContext / cat
        # for training purpose's format
        OutputIntentEvaluation.append("0"+"\t"+""+"\t"+linestrs[0]+"\t" +linestrs[2]+"\t"+linestrs[4]+"\t"+linestrs[5]);

        #UUID\tQuery\tIntent\tDomain\tSlot\r\n
        #OutputSTCAIntentEvaluation.append("0"+"\t"+linestrs[4]+"\t" +linestrs[6]+"\t"+linestrs[5].lower()+"\t"+linestrs[7])

        #id\tquery\tintent\tdomain\tQueryXml\ConversationContext\tcat \r\n"
        OutputSTCAIntentEvaluation.append("0"+"\t"+linestrs[0]+"\t"+linestrs[2]+"\t" +linestrs[1].lower()+"\t"+linestrs[3]+"\t"+linestrs[4]+"\t"+linestrs[5])
        '''

"""
# comment shuffle in the first place
#random.shuffle(OutputSet);
"""


# output soted order for easy check
#outputs = ['\t'.join(['id', 'query', 'intent', 'domain', 'QueryXml'])] + sorted(outputs);
#outputsWithSource = ['\t'.join(['id', 'query', 'intent', 'domain', 'QueryXml', 'source'])] + sorted(outputsWithSource);

#MessageText	JudgedDomain	JudgedIntent	JudgedConstraints	ConversationContext	Cat
outputs = ['\t'.join(['ConversationId','MessageId','MessageTimestamp', 'MessageFrom', 'MessageText', 'JudgedIntent', 'JudgedDomain', 'JudgedConstraints', 'ConversationContext', 'MetaData','Frequency','Cat'])] + sorted(outputs);
outputsWithSource = ['\t'.join(['ConversationId', 'MessageId','MessageTimestamp', 'MessageFrom', 'MessageText','JudgedIntent', 'JudgedDomain', 'JudgedConstraints', 'ConversationContext', 'MetaData', 'Frequency','Cat','source'])] + sorted(outputsWithSource);



with codecs.open(outputFile, 'w', 'utf-8') as fout:
    for item in outputs:
        fout.write(item + '\r\n');


print("dedup size = ")
print(len(outputsSet))
with codecs.open(outputFileUnique, 'w', 'utf-8') as fout:
    fout.write('\t'.join(['ConversationId','MessageId','MessageTimestamp','MessageFrom', 'MessageText', 'JudgedIntent', 'JudgedDomain', 'JudgedConstraints', 'ConversationContext', 'MetaData','Frequency','Cat']) + '\r\n');
    for item in sorted(outputsSet):
        fout.write(item + '\r\n');


print(len(outputsSet))
with codecs.open(outputFileWithSource, 'w', 'utf-8') as fout:
    for item in outputsWithSource:
        fout.write(item + '\r\n');



#with codecs.open(outputFileKeyWordAndFileNameLexiconFilfe, 'w', 'utf-8') as fout:
#    # sort it to easy check
#    for item in sorted(fileKeyWordAndFileNameCandidateSet):
#        fout.write(item + '\r\n');



#with codecs.open(outputFileKeyWordAndFileNameUserContextFiles, 'w', 'utf-8') as fout:
#    # sort it to easy check
#    for item in sorted(outputFileKeyWordAndFileNameUserContext):
#        fout.write(item + '\r\n');


# for CMF slot evaluation format
#with codecs.open((inputFile.split("."))[0] +'slot_evaluation.tsv', 'w', 'utf-8') as fout:
#
#    # if output for traing
#    fout.write("id\tquery\tintent\tdomain\tQueryXml\tConversationContex\tcat\r\n")
#    for item in OutputSlotEvaluation:
#        fout.write(item + '\r\n');
#


'''
# for STCA evaluation
with codecs.open("sharemodeltest\\"+(inputFile.split("."))[0] +'slot_evaluation.tsv', 'w', 'utf-8') as fout:

    # if output for traing
    fout.write("id\tquery\tintent\tdomain\tQueryXml\r\n")
    for item in OutputSlotEvaluation:
        fout.write(item + '\r\n');

# for CMF intent evaluation format
with codecs.open((inputFile.split("."))[0] +'intent_evaluation.tsv', 'w', 'utf-8') as fout:

    # if output for traing
    fout.write("TurnNumber\tPreviousTurnIntent\tquery\tintent\r\n")
    for item in OutputIntentEvaluation:
        fout.write(item + '\r\n');

# for STCAevaluation

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

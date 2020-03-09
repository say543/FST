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


PREVIOUSTURNDOMAIN = "PreviousTurnDomain"
PREVIOUSTURNINTENT = "PreviousTurnIntent"
TASKFRAMESTATUS = "TaskFrameStatus"
TASKFRAMEENTITYSTATES = "TaskFrameEntityStates"
TASKFRAMEGUID = "TaskFrameGUID"
SPEECHPEOPLEDISAMBIGUATIONGRAMMARMATCHES = "SpeechPeopleDisambiguationGrammarMatches"
CONVERSATIONALCONTEXT = "ConversationalContext"


fileDomainRelatedIntent = ['file_search', 'file_open', 'file_share', 'file_download', 'file_other', 'file_navigate', "teamspace_search"]

#borrow from files_mystuff_after_filter.py
'''
fileTypeTagWoDotInFileKeywordOrFileName={

    # space is important
    # order is important

    'pptx' : '<file_type> pptx </file_type> ',
    'ppts' : '<file_type> ppts </file_type> ',
    'ppt' : '<file_type> ppt </file_type> ',
    'deck' : '<file_type> deck </file_type> ',
    'decks' : '<file_type> decks </file_type> ',
    'presentation' : '<file_type> presentation </file_type> ',
    'presentations' : '<file_type> presentations </file_type> ',
    'powerpoint' : '<file_type> powerpoint </file_type> ',
    'PowerPoint' : '<file_type> PowerPoint </file_type> ',
    'powerpoints' : '<file_type> powerpoints </file_type> ',
    # add for seperate
    'power point' : '<file_type> power point </file_type> ',
    
    'slide' : '<file_type> slides </file_type> ',
    'slides' : '<file_type> slides </file_type> ',
    'doc' : '<file_type> doc </file_type> ',
    'docx' : '<file_type> docx </file_type> ',
    'docs' : '<file_type> docs </file_type> ',
    # add for upper case
    'Doc' : '<file_type> Doc </file_type> ',
    'Docx' : '<file_type> Docx </file_type> ',
    'Docs' : '<file_type> Docs </file_type> ',
    # spec no longer being file_type
    #' spec' : '<file_type> spec </file_type> ',
    'excel' : '<file_type> excel </file_type> ',
    'excels' : '<file_type> excels </file_type> ',
    'xls' : '<file_type> xls </file_type> ',
    'xlsx' : '<file_type> xlsx </file_type> ',
    'spreadsheet' : '<file_type> spreadsheet </file_type> ',
    'spreadsheets' : '<file_type> spreadsheets </file_type> ',
    'workbook' : '<file_type> workbook </file_type> ',
    'worksheet' : '<file_type> worksheet </file_type> ',
    'csv' : '<file_type> csv </file_type> ',
    'tsv' : '<file_type> tsv </file_type> ',
    'note' : '<file_type> note </file_type> ',
    'notes' : '<file_type> notes </file_type> ',
    'onenote' : '<file_type> onenote </file_type> ',
    'onenotes' : '<file_type> onenotes </file_type> ',
    # add for upper case
    'OneNote' : '<file_type> OneNote </file_type> ',
    'notebook' : '<file_type> notebook </file_type> ',
    'notebooks' : '<file_type> notebooks </file_type> ',
    'pdf' : '<file_type> pdf </file_type> ',
    'pdfs' : '<file_type> pdfs </file_type> ',
    # add for upper case
    'PDF' : '<file_type> PDF </file_type> ',
    'jpg' : '<file_type> jpg </file_type> ',
    'jpeg' : '<file_type> jpeg </file_type> ',
    'gif' : '<file_type> gif </file_type> ',
    'png' : '<file_type> png </file_type> ',
    'image' : '<file_type> image </file_type> ',
    'msg' : '<file_type> msg </file_type> ',
    'ics' : '<file_type> ics </file_type> ',
    'vcs' : '<file_type> vcs </file_type> ',
    'vsdx' : '<file_type> vsdx </file_type> ',
    'vssx' : '<file_type> vssx </file_type> ',
    'vstx' : '<file_type> vstx </file_type> ',
    'vsdm' : '<file_type> vsdm </file_type> ',
    'vssm' : '<file_type> vssm </file_type> ',
    'vstm' : '<file_type> vstm </file_type> ',
    'vsd' : '<file_type> vsd </file_type> ',
    'vdw' : '<file_type> vdw </file_type> ',
    'vss' : '<file_type> vss </file_type> ',
    'vst' : '<file_type> vst </file_type> ',
    'mpp' : '<file_type> mpp </file_type> ',
    'mpt' : '<file_type> mpt </file_type> ',
    # no mention in spec
    # move it to not tag
    'word' : '<file_type> word </file_type> ',


    # keep it as tag
    'picture' : '<file_type> picture </file_type> ',
    'music' : '<file_type> music </file_type> ',
    'txt' : '<file_type> txt </file_type> ',
}
'''

fileTypeDomanBoost =set([
    'pptx',
    'ppts',
    'ppt',
    'deck',
    'decks',
    'presentation',
    'presentations',
    'powerpoint',
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
    'words',
    'document',
    'documents',
    'file',
    'files'
    ])

#filterDomainDic =set([
#    "mystuff"
#    ])

domainIgnoreList = {}

domainToFileDomain = {
    # carina need small but bellevue needs big
    "mystuff" : "files",
    "MYSTUFF" : "FILES"
}




#OutputSlotEvaluation = [];

#OutputIntentEvaluation = [];


#OutputSTCAIntentEvaluation = [];


domainListDictionary = defaultdict(list);


# validaitng opened dataset
inputFile = "mediacontrol_domain_train.tsv"




isHeadColumn = True
headColumnList =[]
totalRows = 0
with codecs.open(inputFile, 'r', 'utf-8') as fin:
    
    for line in fin:

        #skip headcolumn and check if valid
        if (isHeadColumn):
            line = line.strip();
            if not line:
                continue;
            #headColumnList = line.split('\t');
            #if len(headColumnList) < TARGETNUMCOLUMNS:
            #    print("error header for file: " + file);
                    
            isHeadColumn = False
            continue
        
        line = line.strip();
        if not line:
            continue;
        linestrs = line.split("\t");

        # make sure it at least has
        # Query	ExternalFeature	Weight	Intent	Domain	Slot
        if len(linestrs) < 4:
            continue;

        totalRows+=1
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



print("total valid row\t" + str(totalRows));  

"""
# comment shuffle in the first place
#random.shuffle(OutputSet);
"""

# outout to different bucket for analysis(without anyfilter)
for domain, rows in domainListDictionary.items():
    print("domain\t" + domain);
    print("rows\t" + str(len(rows)));
    with codecs.open((inputFile.split("."))[0] +'_'+domain+'.tsv', 'w', 'utf-8') as fout:
        # header
        #fout.write("TurnNumber\tPreviousTurnDomain\tquery\tdomain\r\n")
        fout.write('\t'.join(['TurnNumber', PREVIOUSTURNINTENT, 'query', 'domain',PREVIOUSTURNDOMAIN, TASKFRAMESTATUS, TASKFRAMEENTITYSTATES, TASKFRAMEGUID, SPEECHPEOPLEDISAMBIGUATIONGRAMMARMATCHES, CONVERSATIONALCONTEXT])+'\r\n');
        for row in rows:
            fout.write(row + '\r\n');


# filter igonre domain list to output

# store ignore data
outputMystuffIgnoreListDuetoFileType = []
outputMyStuffAfterFileTypeFilter= []

# output all domains other than mystuff
# mystuff will filter based on file type
with codecs.open((inputFile.split("."))[0] +'_after_filter'+'.tsv', 'w', 'utf-8') as fout:
    # header
    #fout.write("TurnNumber\tPreviousTurnDomain\tquery\tdomain\r\n")
    fout.write('\t'.join(['TurnNumber', PREVIOUSTURNINTENT, 'query', 'domain',PREVIOUSTURNDOMAIN, TASKFRAMESTATUS, TASKFRAMEENTITYSTATES, TASKFRAMEGUID, SPEECHPEOPLEDISAMBIGUATIONGRAMMARMATCHES, CONVERSATIONALCONTEXT])+'\r\n');
    for domain, lines in domainListDictionary.items():


        # for debug
        #print(domain)
        
        # skip igonre domain based fileTypeDomanBoost
        #if domain.lower() in filterDomainDic:
        if domain.lower() == "mystuff":

            for line in lines:
                line = line.strip();
                if not line:
                    continue;

                # replace all . with \t
                
                linestrs = line.split("\t");
                query = linestrs[2]

                # replace all  ./space/,/?/! with \t
                # not deal with PDF's
                # do it in the future
                
                query = str.replace(query, " ", "\t")
                query = str.replace(query, ".", "\t")
                query = str.replace(query, ",", "\t")
                query = str.replace(query, "?", "\t")
                query = str.replace(query, "!", "\t")

                querytrs = query.split("\t");

                hasFileType = False;
                for querystr in querytrs:
                    if querystr.lower() in fileTypeDomanBoost:
                        hasFileType = True
                        break


                # if really want to use this as domian initial data
                # follow up items
                # ? may be adding file close or end as filter since do not support close xxx or end ... save

                    
                if hasFileType:
                    # store original
                    outputMyStuffAfterFileTypeFilter.append(line)

                    # rename mystuff to files fo rtesting
                    #fout.write(line + '\r\n');
                    fout.write(linestrs[0]+'\t'+linestrs[1]+'\t'+linestrs[2]+'\t'+domainToFileDomain[linestrs[3]]+'\r\n');
                    
                else:
                    outputMystuffIgnoreListDuetoFileType.append(line)

            #print(len(rows))
        else:
            
            for line in lines:
                fout.write(line + '\r\n');

            
# output mystuff queres without file tpye
with codecs.open((inputFile.split("."))[0] +'_mystuff_wo_filter_type'+'.tsv', 'w', 'utf-8') as fout:
    # header
    #fout.write("TurnNumber\tPreviousTurnDomain\tquery\tdomain\r\n")
    fout.write('\t'.join(['TurnNumber', PREVIOUSTURNINTENT, 'query', 'domain',PREVIOUSTURNDOMAIN, TASKFRAMESTATUS, TASKFRAMEENTITYSTATES, TASKFRAMEGUID, SPEECHPEOPLEDISAMBIGUATIONGRAMMARMATCHES, CONVERSATIONALCONTEXT])+'\r\n');
    for item in outputMystuffIgnoreListDuetoFileType:
        fout.write(item + '\r\n');

# output mystuff queres with file tpye
with codecs.open((inputFile.split("."))[0] +'_mystuff_with_filter_type'+'.tsv', 'w', 'utf-8') as fout:
    # header
    #fout.write("TurnNumber\tPreviousTurnDomain\tquery\tdomain\r\n")
    fout.write('\t'.join(['TurnNumber', PREVIOUSTURNINTENT, 'query', 'domain',PREVIOUSTURNDOMAIN, TASKFRAMESTATUS, TASKFRAMEENTITYSTATES, TASKFRAMEGUID, SPEECHPEOPLEDISAMBIGUATIONGRAMMARMATCHES, CONVERSATIONALCONTEXT])+'\r\n');
    for item in outputMyStuffAfterFileTypeFilter:
        fout.write(item + '\r\n');
        

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

    # if output for traingfout.write(item + '\r\n');
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

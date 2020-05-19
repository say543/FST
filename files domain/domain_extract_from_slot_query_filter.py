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



domainToNotSureDomain = {
    # carina need small but bellevue needs big
    "files" : "NOTSURE",
    "FILES" : "NOTSURE"
}

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
    'excel',
    'excels',
    'xls',
    'xlsx',
    'sheet',
    'sheets',
    'spreadsheet',
    'spreadsheets',
    'workbook',
    'worksheet',
    'csv',
    'tsv',
    'onenote',
    'onenotes',
    'onenote',
    'pdf',
    'pdfs',
    'pdf',
    'png',
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

    
OutputSlotEvaluation = [];

OutputIntentEvaluation = [];

OutputSTCAIntentEvaluation = [];


# only files by judge annotation
OutputSpellFilterEvaluationOnlyFiles = [];
OutputSpellFilterEvaluation = [];

# only files by judge annotation, adding file_type filter to correct annotation
OutputSpellFilterEvaluationOnlyFilesDomainFileTypeFilter = [];
OutputSpellFilterEvaluationDomainFileTypeFilter = [];



OutputSpellWrongFilterEvaluation = [];



OutputDomainEvaluation = [];


lexiconSet = set()
with codecs.open('..\\LexiconFiles\\lexicon.calendar.person_names_for_training.txt', 'r', 'utf-8') as fin:
    for line in fin:
        line = line.strip()
        lexiconSet.add(line)

filetypeSet = set()
with codecs.open('..\\lexicons\\file_type_domain_boost_UWP.txt', 'r', 'utf-8') as fin:
    for line in fin:
        line = line.strip()
        filetypeSet.add(line)


inputFile = "domain_extract_from_slot_query.tsv"

#with codecs.open('Teams-golden.tsv', 'r', 'utf-8') as fin:
with codecs.open(inputFile, 'r', 'utf-8') as fin:
    
    for line in fin:
        line = line.strip();
        if not line:
            continue;
        linestrs = line.split("\t");


        # replace all . with \t
                
        query = linestrs[0]

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

        if hasFileType:
            OutputDomainEvaluation.append("0"+"\t\t"+linestrs[0]+"\t"+"files"+"\t\t\t\t\t\t\t"+inputFile);


            
"""
# comment shuffle in the first place
#random.shuffle(OutputSet);
"""

# for judge trainer format
#with codecs.open('teams_golden_after_filtering.tsv', 'w', 'utf-8') as fout:
#
#    # if outout originla format
#    fout.write("ConversationId\tMessageId\tMessageTimestamp\tMessageFrom\tMessageText\tJudgedDomain\tJudgedIntent\tJudgedConstraints\tMetaData\tConversationContext\tFrequency\tImplicitConstraints\r\n")
#    for item in Output:
#        fout.write(item + '\r\n');

# for CMF slot evaluation format
with codecs.open((inputFile.split("."))[0] +'_domain_extraction.tsv', 'w', 'utf-8') as fout:

    # if output for traing
    fout.write("TurnNumber\tPreviousTurnIntent\tquery\tdomain\tPreviousTurnDomain\tTaskFrameStatus\tTaskFrameEntityStates\tTaskFrameGUID\tSpeechPeopleDisambiguationGrammarMatches\tConversationalContext\tSource\r\n")
    
    for item in OutputDomainEvaluation:
        fout.write(item + '\r\n');

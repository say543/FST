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

def readLines(filename):
    output = []
    isHeadColumn = True
    headColumnList =[]
    totalRows = 0
    with codecs.open(filename, 'r', 'utf-8') as fin:
    
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

            output.append(line)
            totalRows+=1

    return output




#inputFile1 = "files_intent_training_checkin.tsv"
#inputFile2 = "files_intent_training.tsv"


inputFile1 = "files_intent_training_checkin.tsv"
inputFile2 = "files_intent_training.tsv"




output1 = readLines(inputFile1)
output2 = readLines(inputFile2)

output1.sort()
output2.sort()





"""
# comment shuffle in the first place
#random.shuffle(OutputSet);
"""
            
# output file1 after sort
with codecs.open((inputFile1.split("."))[0] +'_sort'+'.tsv', 'w', 'utf-8') as fout:
    # header
    fout.write("TurnNumber\tPreviousTurnDomain\tquery\tdomain\r\n")
    for item in output1:
        fout.write(item + '\r\n');

# output file2 after sort
with codecs.open((inputFile2.split("."))[0] +'_sort'+'.tsv', 'w', 'utf-8') as fout:
    # header
    fout.write("TurnNumber\tPreviousTurnDomain\tquery\tdomain\r\n")
    for item in output2:
        fout.write(item + '\r\n');
        


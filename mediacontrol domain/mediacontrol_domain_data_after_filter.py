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
SOURCE= 'Source'

fileDomainRelatedIntent = ['file_search', 'file_open', 'file_share', 'file_download', 'file_other', 'file_navigate', "teamspace_search"]


activeDomain = set([
    'ondevice',
    'reminder',
    'note',
    'alarm',
    'calendar',
    'common',
    'communication',
    'web'
    ])

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
    #'note',
    #'notes',
    'onenote',
    'onenotes',
    'onenote',
    #'notebook',
    #'notebooks',
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

fileTypeDomanBoostRemove =set([
    'note',
    'notes',
    'notebook',
    'notebooks'
    ])

# <microsoft doc/ microsfot excel / microsoft word / window/ windows>

# no folder since folder is mystuff

blackListQuerySet = {
    # regular group
    #excel
    'excel',
    'open up excel',
    'open up excel.',
    'hey cortana open up excel',
    'open up a new excel',
    'open up a new excel.',
    'open excel',
    'open excel.',
    'open new excel',
    'open new excel.',
    'open a new excel',
    'open a new excel.',
    'open excel again',
    'open excel again.',
    'open a new excel again',
    'open a new excel again.',
    'go to excel',
    'take me to excel',
    #word
    'word',
    'open up word',
    'open up word.',
    'hey cortana open up word',
    'open up a new word',
    'open up a new word.',
    'open word',
    'open word.',
    'open new word',
    'open new word.',
    'open a new word',
    'open a new word.',
    'open word again',
    'open word again.',
    'open a new word again',
    'open a new word again.',
    'go to word',
    'take me to word',
    #words
    'words',
    'open up words',
    'open up words.',
    'hey cortana open up words',
    'open up a new words',
    'open up a new words.',
    'open words',
    'open words.',
    'open new words',
    'open new words.',
    'open a new words',
    'open a new words.',
    'open words again',
    'open words again.',
    'open a new words again',
    'open a new words again.',
    'go to words',
    'take me to words',    
    #powerpoint
    'powerpoint',
    'open up powerpoint',
    'open up powerpoint.',
    'hey cortana open up powerpoint',
    'open up a new powerpoint',
    'open up a new powerpoint.',
    'open powerpoint',
    'open powerpoint.',
    'open new powerpoint',
    'open new powerpoint.',
    'open a new powerpoint',
    'open a new powerpoint.',
    'open powerpoint again',
    'open powerpoint again.',
    'open a new powerpoint again',
    'open a new powerpoint again.',
    'go to powerpoint',
    'take me to powerpoint',
    #onenote
    'onenote',
    'open up onenote',
    'open up onenote.',
    'hey cortana open up onenote',
    'open up a new onenote',
    'open up a new onenote.',
    'open onenote',
    'open onenote.',
    'open new onenote',
    'open new onenote.',
    'open a new onenote',
    'open a new onenote.',
    'open onenote again',
    'open onenote again.',
    'open a new onenote again',
    'open a new onenote again.',
    'go to onenote',
    'take me to onenote',
    #spreadsheet
    'spreadsheet',
    'open up spreadsheet',
    'open up spreadsheet.',
    'open up a new spreadsheet',
    'open up a new spreadsheet.',
    'hey cortana open up spreadsheet',
    'open spreadsheet',
    'open spreadsheet.',
    'open new spreadsheet',
    'open new spreadsheet.',
    'open a new spreadsheet',
    'open a new spreadsheet.',
    'open spreadsheet again',
    'open spreadsheet again.',
    'open a new spreadsheet again',
    'open a new spreadsheet again.',
    'go to spreadsheet',
    'take me to spreadsheet'
    # add single query you want in the future
    }






phraseToRemove=set([
    'microsoft doc',
    'microsoft excel',
    'microsoft word',
    'microsoft office',
    'ms word',
    'google',
    'photoshop',
    'file explorer',
    'window',
    'windows',
    'office word',
    'word office',
    'word 2019',
    'word 2016',
    'word 2013',
    'word 2003',
    'word 2000',
    'word 2010',
    'word 2007',
    'word 1996',
    'word app',
    'powerpoint 2019',
    'powerpoint 2016',
    'powerpoint 2013',
    'powerpoint 2003',
    'powerpoint 2000',
    'powerpoint 2010',
    'powerpoint 2007',
    'powerpoint 1996',
    'powerpoint app',
    'excel 2019',
    'excel 2016',
    'excel 2013',
    'excel 2003',
    'excel 2000',
    'excel 2010',
    'excel 2007',
    'excel 1996',
    'excel app',
    'onenote 2019',
    'onenote 2016',
    'onenote 2013',
    'onenote 2003',
    'onenote 2000',
    'onenote 2010',
    'onenote 2007',
    'onenote 1996',
    'onenote app',
    'word book',
    'app',
    'file menu',
    'files menu',
    'system file',
    'system file',
    'word settings',
    'microsoft store',
    'document settings',
    'documents settings',
    'file settings',
    'files settings',
    'program',
    # filter all print related queries as possible queries
    'print',
    # filter all print related queries as possible queries
    'scan',
    # a phrase should be ondevice 
    'Open word for me'
    ])
#filterDomainDic =set([
#    "mystuff"
#    ])

domainToFileDomain = {
    # carina need small but bellevue needs big
    "mystuff" : "files",
    "MYSTUFF" : "FILES",
    "ondevice" : "files",
    "ONDEVICE" : "FILES",
}


# queries might be any domains
# this one has highest priority compare to phrase, key any application
queriesToFiles=set([
    #video file, mystuff treats it as file_type so also add to train
    'Video files.',
    'Video files',
    'Where are my video files',
    'Find video files',
    'Find video file',
    # program file, mystuff treats it as keyword so also add to train
    'open program file',
    'open program files',
    'program files',
    'program file',
    'show program files',
    'show program file',
    'where are my program files',
    # open note, limited cases and extract most mystuff querestion
    'Open note on my computer',
    'Open note for Walbaums grocery',
    'where do you think i can you open note please?',
    'open note with subject...',
    'open note of errands',
    "open note titled brother's name",
    'Open note from yesterday.',
    'Open note with subject of restaurant.',
    'open note love note',
    'open note restaurant',
    'open note regarding',
    'open note from yesterday.',
    # open note,
    'open notes on desktop',
    'open notes for meeting',
    'Open notes for meeting',
    'Open notes about trip',
    'Open notes about grocery shopping',
    "Open notes for my car's issues.",
    'open notes about upcoming trip',
    'open notes for christmas',
    'open notes for chores',
    'open notes about tasks',
    'open notes relating to party',
    # filter negative examples
    ### web domain / deck decks
    'floating tile deck',
    'reddick space deck',
    'bdm deck',
    'smith tower seattle observation deck',
    'north deck',
    'open duel deck',
    'magic event deck',
    'magic kids starter deck',
    'dr deck',
    'budget deck',
    'master deck',
    'amaz priest deck',
    'magic bw heroic deck',
    'hearhstone basiv mage deck',
    'hearthstone standard decks',
    'vintage cassette decks',
    'hearthstone top decks',
    "urza's theme decks",
    'hosoi decks',
    'dixie roof deck'
    ### web domain / doc
    'tom doc',
    'john michael floyd minnesota doc',
    'my word doc',
    'open a word doc',
    # not sure how about this one so keeps it
    #'Open a new word doc',
    #'open a blank word doc',
    'saque fgts doc',
    'movie doc',
    'Jonworksheet.doc',
    'backexercises.doc',
    'word doc',
    'adobe online editing doc',
    ### note domain / doc
    #leave it right know
    #'note send sorat design doc',
    ### ondevice domain / doc
    #leave it right know
    #'Open up word doc',
    ### web domain / doc
    'ted docs',
    ### web domain / spec
    'chevy volt generator spec',
    'uscis green card photi spec',
    'axle nut torque spec',
    's4 lte spec',
    'rdmr spec',
    'surface 2 spec',
    'lumia five twenty full spec',
    'hdmi spec',
    'bluetootg 4.0 spec',
    ### web domain / excel /excels
    #leave it right know
    #open my excel,
    #search excel,
    #opening excel,
    #find excel,
    #looking for excel
    # a very big bucket for web
    #hey cortana open excel,
    #search up excel,
    'nfl 2013 schedule printable excel',
    'self teach excel',
    'depth chart nfl defense excel',
    'nfl schedule 2013 excel',
    'nfl weekly schedule 2012 excel',
    'daily journal template excel',
    '2013 nfl monday night football schedule excel',
    'nfl schedule 2012 excel',
    'i excel',
    'dart excel',
    'calendar server excel',
    'watt pad word excel',
    ### web domain xls
    'nfl schedule 2013 xls',
    ### ondeivce domain / spreadsheet
    # leave it only one
    #xcel spreadsheet
    ### web domain spreadsheet
    # leave it right know
    #Open a new excel spreadsheet
    'cell spreadsheet',
    '2011-2012 tax spreadsheet',
    'Daniel`s spreadsheet',
    'cortanta cntl p spreadsheet',
    '2012 nfl schedule spreadsheet',
    ### web domain workbook
    'philosophy workbook',
    ### web domain worksheet
    'time management matrix worksheet',
    'find discussion worksheets',
    'teacher worksheets',
    # leave it now
    #reading worksheets'
    'math worksheets',
    'hey henneke worksheets',
    'food chains and food webs worksheets',
    'hey cora hanako worksheets again',
    'and most data xl worksheets',
    'dave ramsey worksheets',
    'addition worksheets',
    'fun worksheets',
    'baseball depth chart worksheet',
    'common core worksheets',
    'Jon worksheet',
    'blood worksheet',
    'so security benefits worksheet',
    'Jonworksheet.doc',
    'adding and subtracting rational expressions worksheet',
    ### web domain csv
    'nfl 2013 schedule csv',
    'fit to csv',
    ### web domain pdf
    # # leave it now
    # open pdf
    #'change default pdf',
    'ge annual report 2013 pdf',
    '2013-2014 denver broncos schedule pdf',
    'nfl football schedule 2012 pdf',
    'big climb 2013 results pdf',
    'wood carving tutorial pdf',
    'shake and bake pdf',
    'doctor who magazine pdf',
    'nfl schedule 2012 regular season pdf',
    'cah pdf',
    'monofusion pdf',
    'departures pdf',
    'seattle seahawks 2013 schedule pdf',
    'missionary handbook lds pdf',
    'professional software development pdf',
    'thesis pdf',
    'nfl league schedule 2012 pdf',
    'search for comic pdf',
    'contractor lien release pdf',
    'open receipt dot pdf',
    'the last question pdf',
    'missionary handbook pdf',
    'oecd model pdf',
    'shake and bake meth recipe pdf',
    'brian sent selection procedures pdf',
    'topo podlesice pdf',
    'octave manual pdf',
    'callister pdf',
    'g doll using pdf',
    'nfl sunday ticket schedule pdf',
    'calibre pdf',
    'chicago bulls schedule 2013 2014 pdf',
    'dallas cowboys 2013 schedule pdf',
    'skydrive picture pdf',
    'download pdf',
    'art of computer programming pdf',
    'drawboard pdf',
    'nfl week 4 schedule pdf',
    'duty to god pdf',
    'crucial conversations pdf',
    'nfl schedule 2014 pdf',
    'steelers schedule 2013 pdf',
    'find six glasses pdf',
    'meth growing cookbook pdfeth growing cookbook pdf',
    'nuwave oven book recipe pdf',
    ### web domain jpeg
    # not sure if turn on so leave it 
    #leave it now
    ##'define jpeg web'
    #'facebook for me dot jpeg',
    #'lg motion jpeg',
    #'stenographic jpeg',
    ### web domain image
    ### leave
    ###can you find the mobile phone image
    ### not finish, do it in the futrue
    #'mh370 satellite image',
    #'large smiley image',
    #'anthony tony schwarz image',
    #'search image',
    #'search for image',
    #'creme  donut image',
    #'fish image',
    #'search by image',
    #'two thousand fifteen camaro image',
    #'slay one image',
    #'cortana find me a sword bubble image',
    #'creeper image',
    #'muse paint image'
    ### web domain msg
    'nutritional yeast msg',
    'inchin bamboo garden msg',
    ### web domain word
    # mant data not file name so leave it
    # not it finish all
    # leave it
    # learn a word
    #'is dunk a word',
    #'is calling a word',
    #'translate the japanese word',
    #open up word,
    'open ellis word',
    ### web domain words
    # mant data not file name so leave it
    # not it finish all
    'find croatian words',
    'find words',
    'jumble words',
    'lut words',
    ### web domain document
    # leave it
    #open new document
    #open up my current word document
    #open new word document
    #open document
    'word document',
    'working document',
    'sample pdf document',
    'a loan document',
    'open a text document',
    'stamp document',
    'school document',
    'search document',
    'search file hosting for document',
    'a word document',
    'open secret document',
    'open a powerpoint document',
    ### web domain documents
    ## leave it
    ## open for me the documents
    ##video documents
    ##cortana open my documents
    ##hey cortana open up documents
    ##find documents
    "pdf documents",
    'bing documents',
    "Sister store documents",
    'open petrobras documents',
    'h. four documents',
    'go court documents',
    'quickflight documents',
    'forge legal documents',
    'open old publisher documents'
    ### web domain file
    # leave it
    #one file
    #open file
    #the it crest file
    #windows ten technical review iso file
    #'surface pro four recovery file',
    #open up a new file
    #htc one mini file
    #onedrive file
    #administration file
    #word file
    #sign file
    #picture file
    #search outlook template file
    'ucs file',
    'find me a forex file',
    'dave ray golf tips file',
    'lol file',
    'heat map file',
    'playbook file',
    'like file',
    'games file',
    'bluestacks setup file',
    'cortana open my ucs file',
    'open finance file',
    'vhd file',
    'hey cortana open trust file',
    'laura file',
    'Candle Shop Expense File',
    'synology ds file',
    'troubleshoot advanced threat analytics with blood file',
    'showbox apx file',
    'timesheet three file'
    'find gucci mane world war three on lean file',
    'jurassic garden file',
    'oscar open file',
    'download sound file',
    'chm file',
    'download pdf file',
    'Seasonal items file',
    'open a .y file',
    'show me powerpoint file',
    'java script file',
    'boy c file',
    'cajun file',
    'amazon magazin file',
    'easy card file',
    ### web domain files
    #can you open up files
    #speech recognition installation files
    #call files
    #allan smith's outlook files
    #open user files,
    #find downloaded music files
    #howdy x files
    #open files,
    #documents files
    #cmd files,
    #find dead files
    #music files,
    #show files
    #search video files
    #torrent files,
    #all files
    #find zip files
    'verizon files',
    'mp four files',
    'show me some episodes of the dead files',
    'the enigma files',
    'yamaha rx fifteen original load files',
    'the rockford files',
    'search lml holding llc files',
    'download files',
    'mixed up files',
    'show me pdf files',
    'od files',
    'itunes files',
    'please open pdf files',
    'employment files',
    'your level three play files',
    'show me my music files',
    'nancy drew files',
    'titanfall sound files',
    'hey cortana show me stats files',
    'download my files',
    'text files',
    'ussr secret alien files',
    'mkv files',
    'conspiracy files',
    'tobolowsky files',
    'dresden files',
    'show me all of your pdf files',
    'wchd dot org user files',
    'mindmaple files',
    'minneapolis local files',
    'jillian anderson x files',
    'search gym butcher dresden files',
    'gate files',
    'find me fallout three files'
    ])

# merge into above 
#WebQuerySetUpdateDomain = set([
#    'floating tile deck',
#    'reddick space deck',
#    'bdm deck',
#    'smith tower seattle observation deck',
#    'north deck',
#    'open duel deck',
#    'magic event deck',
#    'magic kids starter deck',
#    'dr deck',
#    'budget deck',
#    'master deck',
#    'amaz priest deck',
#    'magic bw heroic deck',
#    'hearhstone basiv mage deck',
#    ])


queriesToFilesLowCase = set(query.lower() for query in queriesToFiles)
# for debug
#print(queriesToFilesLowCase)



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
        fout.write('\t'.join(['TurnNumber', PREVIOUSTURNINTENT, 'query', 'domain',PREVIOUSTURNDOMAIN, TASKFRAMESTATUS, TASKFRAMEENTITYSTATES, TASKFRAMEGUID, SPEECHPEOPLEDISAMBIGUATIONGRAMMARMATCHES, CONVERSATIONALCONTEXT, SOURCE])+'\r\n');
        for row in rows:
            fout.write(row + '\r\n');


# filter igonre domain list to output

# store ignore data
outputMystuffIgnoreListDuetoFileType = []
outputMyStuffAfterFileTypeFilter= []

outputOndeviceIgnoreListDuetoFileType = []
outputOndeviceAfterFileTypeFilter= []


# output all domains other than mystuff
# mystuff will filter based on file type
with codecs.open((inputFile.split("."))[0] +'_after_filter'+'.tsv', 'w', 'utf-8') as fout:
    # header
    #fout.write("TurnNumber\tPreviousTurnDomain\tquery\tdomain\r\n")
    fout.write('\t'.join(['TurnNumber', PREVIOUSTURNINTENT, 'query', 'domain',PREVIOUSTURNDOMAIN, TASKFRAMESTATUS, TASKFRAMEENTITYSTATES, TASKFRAMEGUID, SPEECHPEOPLEDISAMBIGUATIONGRAMMARMATCHES, CONVERSATIONALCONTEXT, SOURCE])+'\r\n');
    for domain, lines in domainListDictionary.items():


        # for debug
        #print(domain)
        
        # skip igonre domain based fileTypeDomanBoost
        #if domain.lower() in filterDomainDic:
        # ? extend to ondevice snice it has many word document
        # in the future word document  in mystuff or ondevice both exists
        # if all mystuff then no need to have ondevice branch
        if domain.lower() == "mystuff":

            for line in lines:
                line = line.strip();
                if not line:
                    continue;

                # replace all . with \t
                
                linestrs = line.split("\t");
                originalQuery = linestrs[2]
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
                    if querystr.lower() in reversed(sorted(fileTypeDomanBoost)):
                        hasFileType = True
                        # no early terminate and keep checking
                        #break;

                    # if one in remove list, set it false and early terminate
                    if querystr.lower() in fileTypeDomanBoostRemove:
                        hasFileType = False
                        break;

                hasphraseToRemove = False
                for key in reversed(sorted(phraseToRemove)):

                    if (originalQuery.lower().find(key)) != -1:
                        hasphraseToRemove = True
                        #if (key == 'microsoft word'):
                        #    print(originalQuery)
                        break


                skip = False
                # cannot use str
                # it will override existing function
                # ? this can be improved by using set for blacklistQuery and o(1) compare
                for blackListQuery in sorted (blackListQuerySet) :            
                    if originalQuery.lower() == blackListQuery.lower():
                        #print(originalQuery)
                        skip = True
                        break
                   

                # if really want to use this as domian initial data
                # follow up items
                # ? may be adding file close or end as filter since do not support close xxx or end ... save

                if originalQuery.lower() in queriesToFilesLowCase:
                    # store original
                    outputMyStuffAfterFileTypeFilter.append(line+'\t\t\t\t\t\t\t'+inputFile)
                    # rename mystuff to files fo rtesting
                    #fout.write(line + '\r\n');
                    fout.write(linestrs[0]+'\t'+linestrs[1]+'\t'+linestrs[2]+'\t'+'files'+'\t\t\t\t\t\t\t'+inputFile+'r\n');
                elif hasFileType:
                    # store original
                    outputMyStuffAfterFileTypeFilter.append(line+'\t\t\t\t\t\t\t'+inputFile)

                    if hasphraseToRemove is False and skip is False:
                        # rename mystuff to files fo rtesting
                        #fout.write(line + '\r\n');
                        fout.write(linestrs[0]+'\t'+linestrs[1]+'\t'+linestrs[2]+'\t'+domainToFileDomain[linestrs[3]]+'\t\t\t\t\t\t\t'+inputFile+'r\n');
                    #else:
                        # not append as negative scenarios since some mystuff needs to support
                        # like folder
                    #    fout.write(linestrs[0]+'\t'+linestrs[1]+'\t'+linestrs[2]+'\t'+linestrs[3]+'\t\t\t\t\t\t\t'+inputFile+'r\n');
                else:
                    outputMystuffIgnoreListDuetoFileType.append(line+'\t\t\t\t\t\t\t'+inputFile)

            #print(len(rows))

        elif domain.lower() == "ondevice":

            for line in lines:
                line = line.strip();
                if not line:
                    continue;

                # replace all . with \t
                
                linestrs = line.split("\t");
                originalQuery = linestrs[2]
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
                    if querystr.lower() in reversed(sorted(fileTypeDomanBoost)):
                        hasFileType = True
                        # no early terminate and keep checking
                        #break;

                    # if one in remove list, set it false and early terminate
                    if querystr.lower() in fileTypeDomanBoostRemove:
                        hasFileType = False
                        break;

                hasphraseToRemove = False
                for key in reversed(sorted(phraseToRemove)):

                    #print (key)
                    #print (query.lower())
                    #print (query.lower().find(key))
                    if (originalQuery.lower().find(key)) != -1:
                        hasphraseToRemove = True
                        #if (key == 'microsoft word'):
                        #    print(originalQuery)
                        break

                skip = False
                # cannot use str
                # it will override existing function
                # ? this can be improved by using set for blacklistQuery and o(1) compare
                for blackListQuery in sorted (blackListQuerySet) :            
                    if originalQuery.lower() == blackListQuery.lower():
                        #print(originalQuery)
                        skip = True
                        break



                # if really want to use this as domian initial data
                # follow up items
                # ? may be adding file close or end as filter since do not support close xxx or end ... save

                if originalQuery.lower() in queriesToFilesLowCase:
                    # store original
                    outputOndeviceAfterFileTypeFilter.append(line+'\t\t\t\t\t\t\t'+inputFile)
                    # rename ondeice to files fo rtesting
                    #fout.write(line + '\r\n');
                    fout.write(linestrs[0]+'\t'+linestrs[1]+'\t'+linestrs[2]+'\t'+'files'+'\t\t\t\t\t\t\t'+inputFile+'r\n');
                if hasFileType:
                    # store original
                    outputOndeviceAfterFileTypeFilter.append(line+'\t\t\t\t\t\t\t'+inputFile)

                    if hasphraseToRemove is False and skip is False:
                        # rename ondevice to files fo rtesting
                        #fout.write(line + '\r\n');
                        fout.write(linestrs[0]+'\t'+linestrs[1]+'\t'+linestrs[2]+'\t'+domainToFileDomain[linestrs[3]]+'\t\t\t\t\t\t\t'+inputFile+'r\n');
                    else:
                        fout.write(linestrs[0]+'\t'+linestrs[1]+'\t'+linestrs[2]+'\t'+linestrs[3]+'\t\t\t\t\t\t\t'+inputFile+'r\n');
                    
                else:
                    outputOndeviceIgnoreListDuetoFileType.append(line+'\t\t\t\t\t\t\t'+inputFile)

            #print(len(rows))


        #duplcate into queriesToFilesLowCase
        #elif domain.lower() in 'web':
        #    for line in lines:
        #        if not line:
        #            continue;

        #        linestrs = line.split("\t");
        #        originalQuery = linestrs[2]

        #        if originalQuery.lower() in WebQuerySetUpdateDomain:
        #            # not sure which domain so do not store original
        #            #outputOndeviceAfterFileTypeFilter.append(line+'\t\t\t\t\t\t\t'+inputFile)
        #            # rename xxx domain to files fo rtesting
        #            #fout.write(line + '\r\n');
        #            fout.write(linestrs[0]+'\t'+linestrs[1]+'\t'+linestrs[2]+'\t'+'files'+'\t\t\t\t\t\t\t'+inputFile+'r\n');
        #        else:
        #            fout.write(line +'\t\t\t\t\t\t\t'+inputFile+'\r\n');
        #        
        #    
        # only add actie domains    
        #else:
        elif domain.lower() in activeDomain:
            
            for line in lines:

                if not line:
                    continue;

                linestrs = line.split("\t");
                originalQuery = linestrs[2]
                
                if originalQuery.lower() in queriesToFilesLowCase:
                    # not sure which domain so do not store original
                    #outputOndeviceAfterFileTypeFilter.append(line+'\t\t\t\t\t\t\t'+inputFile)
                    # rename xxx domain to files fo rtesting
                    #fout.write(line + '\r\n');
                    fout.write(linestrs[0]+'\t'+linestrs[1]+'\t'+linestrs[2]+'\t'+'files'+'\t\t\t\t\t\t\t'+inputFile+'r\n');
                else:
                    fout.write(line +'\t\t\t\t\t\t\t'+inputFile+'\r\n');



# output mystuff queres with file tpye
with codecs.open((inputFile.split("."))[0] +'_mystuff_with_filter_type'+'.tsv', 'w', 'utf-8') as fout:
    # header
    #fout.write("TurnNumber\tPreviousTurnDomain\tquery\tdomain\r\n")
    fout.write('\t'.join(['TurnNumber', PREVIOUSTURNINTENT, 'query', 'domain',PREVIOUSTURNDOMAIN, TASKFRAMESTATUS, TASKFRAMEENTITYSTATES, TASKFRAMEGUID, SPEECHPEOPLEDISAMBIGUATIONGRAMMARMATCHES, CONVERSATIONALCONTEXT, SOURCE])+'\r\n');
    for item in outputMyStuffAfterFileTypeFilter:
        fout.write(item + '\r\n');



# output mystuff queres without file tpye
with codecs.open((inputFile.split("."))[0] +'_mystuff_wo_filter_type'+'.tsv', 'w', 'utf-8') as fout:
    # header
    #fout.write("TurnNumber\tPreviousTurnDomain\tquery\tdomain\r\n")
    fout.write('\t'.join(['TurnNumber', PREVIOUSTURNINTENT, 'query', 'domain',PREVIOUSTURNDOMAIN, TASKFRAMESTATUS, TASKFRAMEENTITYSTATES, TASKFRAMEGUID, SPEECHPEOPLEDISAMBIGUATIONGRAMMARMATCHES, CONVERSATIONALCONTEXT, SOURCE])+'\r\n');
    for item in outputMystuffIgnoreListDuetoFileType:
        fout.write(item + '\r\n');



# output ondevice queres with file tpye
with codecs.open((inputFile.split("."))[0] +'_ondevice_with_filter_type'+'.tsv', 'w', 'utf-8') as fout:
    # header
    #fout.write("TurnNumber\tPreviousTurnDomain\tquery\tdomain\r\n")
    fout.write('\t'.join(['TurnNumber', PREVIOUSTURNINTENT, 'query', 'domain',PREVIOUSTURNDOMAIN, TASKFRAMESTATUS, TASKFRAMEENTITYSTATES, TASKFRAMEGUID, SPEECHPEOPLEDISAMBIGUATIONGRAMMARMATCHES, CONVERSATIONALCONTEXT, SOURCE])+'\r\n');
    for item in outputOndeviceAfterFileTypeFilter:
        fout.write(item + '\r\n');


# output ondevice queres without file tpye
with codecs.open((inputFile.split("."))[0] +'_ondevice_wo_filter_type'+'.tsv', 'w', 'utf-8') as fout:
    # header
    #fout.write("TurnNumber\tPreviousTurnDomain\tquery\tdomain\r\n")
    fout.write('\t'.join(['TurnNumber', PREVIOUSTURNINTENT, 'query', 'domain',PREVIOUSTURNDOMAIN, TASKFRAMESTATUS, TASKFRAMEENTITYSTATES, TASKFRAMEGUID, SPEECHPEOPLEDISAMBIGUATIONGRAMMARMATCHES, CONVERSATIONALCONTEXT, SOURCE])+'\r\n');
    for item in outputOndeviceIgnoreListDuetoFileType:
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

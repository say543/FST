import glob;
import codecs;
import random;
import os
from shutil import copyfile


fileTypeTagWoDotInFileKeywordOrFileName={

    # space is important
    # order is important

    ' pptx' : '<file_type> pptx </file_type> ',
    ' ppts' : '<file_type> ppts </file_type> ',
    ' ppt' : '<file_type> ppt </file_type> ',
    ' deck' : '<file_type> deck </file_type> ',
    ' decks' : '<file_type> decks </file_type> ',
    ' presentation' : '<file_type> presentation </file_type> ',
    ' presentations' : '<file_type> presentations </file_type> ',
    ' powerpoint' : '<file_type> powerpoint </file_type> ',
    ' powerpoints' : '<file_type> powerpoints </file_type> ',
    # add for seperate
    ' power point' : '<file_type> power point </file_type> ',
    
    ' slide' : '<file_type> slides </file_type> ',
    ' slides' : '<file_type> slides </file_type> ',
    ' doc' : '<file_type> doc </file_type> ',
    ' docx' : '<file_type> docx </file_type> ',
    ' docs' : '<file_type> docs </file_type> ',
    # add for upper case
    ' Doc' : '<file_type> Doc </file_type> ',
    ' Docx' : '<file_type> Docx </file_type> ',
    ' Docs' : '<file_type> Docs </file_type> ',

    ' spec' : '<file_type> spec </file_type> ',
    ' excel' : '<file_type> excel </file_type> ',
    ' excels' : '<file_type> excels </file_type> ',
    ' xls' : '<file_type> xls </file_type> ',
    ' xlsx' : '<file_type> xlsx </file_type> ',
    ' spreadsheet' : '<file_type> spreadsheet </file_type> ',
    ' spreadsheets' : '<file_type> spreadsheets </file_type> ',
    ' workbook' : '<file_type> workbook </file_type> ',
    ' worksheet' : '<file_type> worksheet </file_type> ',
    ' csv' : '<file_type> csv </file_type> ',
    ' tsv' : '<file_type> tsv </file_type> ',
    ' note' : '<file_type> note </file_type> ',
    ' notes' : '<file_type> notes </file_type> ',
    ' onenote' : '<file_type> onenote </file_type> ',
    ' onenotes' : '<file_type> onenotes </file_type> ',
    # add for upper case
    ' OneNote' : '<file_type> OneNote </file_type> ',
    ' notebook' : '<file_type> notebook </file_type> ',
    ' notebooks' : '<file_type> notebooks </file_type> ',
    ' pdf' : '<file_type> pdf </file_type> ',
    ' pdfs' : '<file_type> pdfs </file_type> ',
    # add for upper case
    ' PDF' : '<file_type> PDF </file_type> ',
    ' jpg' : '<file_type> jpg </file_type> ',
    ' jpeg' : '<file_type> jpeg </file_type> ',
    ' gif' : '<file_type> gif </file_type> ',
    ' png' : '<file_type> png </file_type> ',
    ' image' : '<file_type> image </file_type> ',
    ' msg' : '<file_type> msg </file_type> ',
    ' ics' : '<file_type> ics </file_type> ',
    ' vcs' : '<file_type> vcs </file_type> ',
    ' vsdx' : '<file_type> vsdx </file_type> ',
    ' vssx' : '<file_type> vssx </file_type> ',
    ' vstx' : '<file_type> vstx </file_type> ',
    ' vsdm' : '<file_type> vsdm </file_type> ',
    ' vssm' : '<file_type> vssm </file_type> ',
    ' vstm' : '<file_type> vstm </file_type> ',
    ' vsd' : '<file_type> vsd </file_type> ',
    ' vdw' : '<file_type> vdw </file_type> ',
    ' vss' : '<file_type> vss </file_type> ',
    ' vst' : '<file_type> vst </file_type> ',
    ' mpp' : '<file_type> mpp </file_type> ',
    ' mpt' : '<file_type> mpt </file_type> ',
    # no mention in spec
    # move it to not tag
    #' word' : '<file_type> word </file_type> ',


    # keep it as tag
    ' picture' : '<file_type> picture </file_type> ',
    ' music' : '<file_type> music </file_type> ',
}


fileTypeTagWDotInFileKeywordOrFileName={

    # space is important
    # order is important

    '.ppt' : '.<file_type> ppt </file_type> ',
    '.pptx' : '.<file_type> pptx </file_type> ',
    '.ppts' : '.<file_type> ppts </file_type> ',
    '.deck' : '.<file_type> deck </file_type> ',
    '.decks' : '.<file_type> decks </file_type> ',
    '.presentation' : '.<file_type> presentation </file_type> ',
    '.presentations' : '.<file_type> presentations </file_type> ',
    '.powerpoint' : '.<file_type> powerpoint </file_type> ',
    '.powerpoints' : '.<file_type> powerpoints </file_type> ',
    # add for seperate
    '.power point' : '.<file_type> power point </file_type> ',
    
    '.slide' : '.<file_type> slides </file_type> ',
    '.slides' : '.<file_type> slides </file_type> ',
    '.doc' : '.<file_type> doc </file_type> ',
    '.docx' : '.<file_type> docx </file_type> ',
    '.docs' : '.<file_type> docs </file_type> ',
    # add for upper case
    '.Doc' : '.<file_type> Doc </file_type> ',
    '.Docx' : '.<file_type> Docx </file_type> ',
    '.Docs' : '.<file_type> Docs </file_type> ',

    
    '.spec' : '.<file_type> spec </file_type> ',
    '.excel' : '.<file_type> excel </file_type> ',
    '.excels' : '.<file_type> excels </file_type> ',
    '.xls' : '.<file_type> xls </file_type> ',
    '.xlsx' : '.<file_type> xlsx </file_type> ',
    '.spreadsheet' : '.<file_type> spreadsheet </file_type> ',
    '.spreadsheets' : '.<file_type> spreadsheets </file_type> ',
    '.workbook' : '.<file_type> workbook </file_type> ',
    '.worksheet' : '.<file_type> worksheet </file_type> ',
    '.csv' : '.<file_type> csv </file_type> ',
    '.tsv' : '.<file_type> tsv </file_type> ',
    '.note' : '.<file_type> note </file_type> ',
    '.notes' : '.<file_type> notes </file_type> ',
    '.onenote' : '.<file_type> onenote </file_type> ',
    '.onenotes' : '.<file_type> onenotes </file_type> ',
    # add for upper case
    '.OneNote' : '.<file_type> OneNote </file_type> ',
    '.notebook' : '.<file_type> notebook </file_type> ',
    '.notebooks' : '.<file_type> notebooks </file_type> ',
    '.pdf' : '.<file_type> pdf </file_type> ',
    # add for upper case
    '.PDF' : '.<file_type> PDF </file_type> ',
    '.pdfs' : '.<file_type> pdfs </file_type> ',
    '.jpg' : '.<file_type> jpg </file_type> ',
    '.jpeg' : '.<file_type> jpeg </file_type> ',
    '.gif' : '.<file_type> gif </file_type> ',
    '.png' : '.<file_type> png </file_type> ',
    '.image' : '.<file_type> image </file_type> ',
    '.msg' : '.<file_type> msg </file_type> ',
    '.ics' : '.<file_type> ics </file_type> ',
    '.vcs' : '.<file_type> vcs </file_type> ',
    '.vsdx' : '.<file_type> vsdx </file_type> ',
    '.vssx' : '.<file_type> vssx </file_type> ',
    '.vstx' : '.<file_type> vstx </file_type> ',
    '.vsdm' : '.<file_type> vsdm </file_type> ',
    '.vssm' : '.<file_type> vssm </file_type> ',
    '.vstm' : '.<file_type> vstm </file_type> ',
    '.vsd' : '.<file_type> vsd </file_type> ',
    '.vdw' : '.<file_type> vdw </file_type> ',
    '.vss' : '.<file_type> vss </file_type> ',
    '.vst' : '.<file_type> vst </file_type> ',
    '.mpp' : '.<file_type> mpp </file_type> ',
    '.mpt' : '.<file_type> mpt </file_type> ',
    # no mention in spec
    # move it to not tag
    #'.word' : '.<file_type> word </file_type> ',

    # keep it as tag
    '.picture' : '.<file_type> picture </file_type> ',
    '.music' : '.<file_type> music </file_type> ',
}

fileTypeTagWDotSpaceInFileKeywordOrFileName={

    # space is important
    # order is important

    '. ppt' : '. <file_type> ppt </file_type> ',
    '. pptx' : '. <file_type> pptx </file_type> ',
    '. ppts' : '. <file_type> ppts </file_type> ',
    '. deck' : '. <file_type> deck </file_type> ',
    '. decks' : '. <file_type> decks </file_type> ',
    '. presentation' : '. <file_type> presentation </file_type> ',
    '. presentations' : '. <file_type> presentations </file_type> ',
    '. powerpoint' : '. <file_type> powerpoint </file_type> ',
    '. powerpoints' : '. <file_type> powerpoints </file_type> ',
    # add for seperate
    '. power point' : '. <file_type> power point </file_type> ',
    '. slide' : '. <file_type> slides </file_type> ',
    '. slides' : '. <file_type> slides </file_type> ',
    '. doc' : '. <file_type> doc </file_type> ',
    '. docx' : '. <file_type> docx </file_type> ',
    '. docs' : '. <file_type> docs </file_type> ',
    # add for upper case
    '. Doc' : '. <file_type> Doc </file_type> ',
    '. Docx' : '. <file_type> Docx </file_type> ',
    '. Docs' : '. <file_type> Docs </file_type> ',

    '. spec' : '. <file_type> spec </file_type> ',
    '. excel' : '. <file_type> excel </file_type> ',
    '. excels' : '. <file_type> excels </file_type> ',
    '. xls' : '. <file_type> xls </file_type> ',
    '. xlsx' : '. <file_type> xlsx </file_type> ',
    '. spreadsheet' : '. <file_type> spreadsheet </file_type> ',
    '. spreadsheets' : '. <file_type> spreadsheets </file_type> ',
    '. workbook' : '. <file_type> workbook </file_type> ',
    '. worksheet' : '. <file_type> worksheet </file_type> ',
    '. csv' : '. <file_type> csv </file_type> ',
    '. tsv' : '. <file_type> tsv </file_type> ',
    '. note' : '. <file_type> note </file_type> ',
    '. notes' : '. <file_type> notes </file_type> ',
    '. onenote' : '. <file_type> onenote </file_type> ',
    '. onenotes' : '. <file_type> onenotes </file_type> ',
    # add for upper case
    '. OneNote' : '. <file_type> OneNote </file_type> ',
    '. notebook' : '. <file_type> notebook </file_type> ',
    '. notebooks' : '. <file_type> notebooks </file_type> ',
    '. pdf' : '. <file_type> pdf </file_type> ',
    '. pdfs' : '. <file_type> pdfs </file_type> ',
    # add for upper case
    '. PDF' : '. <file_type> PDF </file_type> ',    
    '. jpg' : '. <file_type> jpg </file_type> ',
    '. jpeg' : '. <file_type> jpeg </file_type> ',
    '. gif' : '. <file_type> gif </file_type> ',
    '. png' : '. <file_type> png </file_type> ',
    '. image' : '. <file_type> image </file_type> ',
    '. msg' : '. <file_type> msg </file_type> ',
    '. ics' : '. <file_type> ics </file_type> ',
    '. vcs' : '. <file_type> vcs </file_type> ',
    '. vsdx' : '. <file_type> vsdx </file_type> ',
    '. vssx' : '. <file_type> vssx </file_type> ',
    '. vstx' : '. <file_type> vstx </file_type> ',
    '. vsdm' : '. <file_type> vsdm </file_type> ',
    '. vssm' : '. <file_type> vssm </file_type> ',
    '. vstm' : '. <file_type> vstm </file_type> ',
    '. vsd' : '. <file_type> vsd </file_type> ',
    '. vdw' : '. <file_type> vdw </file_type> ',
    '. vss' : '. <file_type> vss </file_type> ',
    '. vst' : '. <file_type> vst </file_type> ',
    '. mpp' : '. <file_type> mpp </file_type> ',
    '. mpt' : '. <file_type> mpt </file_type> ',
    # no mention in spec

    # move it to not tag
    #'. word' : '. <file_type> word </file_type> ',

    # keep it as tag
    '. picture': '. <file_type> picture </file_type> ',
    '. music': '. <file_type> music </file_type> ',
}

outputs = []
outputsFilter = []

with codecs.open('file_name.txt', 'r', 'utf-8') as fin:
    for line in fin:
        line = line.strip();
        ignore = False
        if not ignore:
            for key in reversed(sorted(fileTypeTagWoDotInFileKeywordOrFileName)):
                if line.endswith(key):
                    ignore = True
        if not ignore:
            for key in reversed(sorted(fileTypeTagWDotInFileKeywordOrFileName)):
                if line.endswith(key):
                    ignore = True
        if not ignore:
            for key in reversed(sorted(fileTypeTagWDotSpaceInFileKeywordOrFileName)):
                if line.endswith(key):
                    ignore = True  

        if not ignore:
            outputs.append(line);
        else:
            outputsFilter.append(line)


with codecs.open("file_keyword.txt", 'w', 'utf-8') as fout:
    for item in outputs:
        fout.write(item + '\r\n');

with codecs.open("file_name_filter.txt", 'w', 'utf-8') as fout:
    for item in outputsFilter:
        fout.write(item + '\r\n');

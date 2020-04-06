import codecs;
import random;
import re;


import math
import re
import sys

# add hyper paramter if unbalanced
hyper_parameter = 200



fileDomainRelatedIntent = ['file_search', 'file_open', 'file_share', 'file_download', 'file_other', 'file_navigate', "teamspace_search"]

teamsDomainToFileDomain = {
    "teams" : "files"
}



# miss <action context> and it will need to check manually
teamsSlotToFileSlot = {
    # preprocessing keyword or file_name later
    "<teamspace_searchkeyword>" : "<file_keyword>",
    "</teamspace_searchkeyword>" : "</file_keyword>",
    # preprocessing keyword or file_name later
    "<file_title>" : "<file_keyword>",
    "</file_title>" : "</file_keyword>",
    # preprocessing time or date later
    "<teammeeting_starttime>" : "<meeting_starttime>",
    "</teammeeting_starttime>" : "</meeting_starttime>",
    "<file_filetype>" : "<file_type>",
    "</file_filetype>" : "</file_type>",
    "<file_filetype>" : "<file_type>",
    "</file_filetype>" : "</file_type>",
    "<file_filerecency>" : "<file_recency>",
    "</file_filerecency>" : "</file_recency>",
    # unopned to judge but data has it so cover
    "<file_orderref>" : "<order_ref>",
    "</file_orderref>" : "</order_ref>",
    "<file_sharetarget>" : "<sharetarget_type>",
    "</file_sharetarget>" : "</sharetarget_type>",
    # preprocessing channel or team title later
    "<teammeeting_title>" : "<sharetarget_name>",
    "</teammeeting_title>" : "</sharetarget_name>",
    "<teamspace_channel>" : "<sharetarget_name>",
    "</teamspace_channel>" : "</sharetarget_name>",
    "<teamspace_team>" : "<sharetarget_name>",
    "</teamspace_team>" : "</sharetarget_name>",
    # preprocessing contact name or to contact name later
    "<teamsuser_contactname>" : "<contact_name>",
    "</teamsuser_contactname>" : "</contact_name>",
    "<deck_location> " : "",
    # based on sorting order, having space at the end will be checked at first
    # so check space version then no space
    "</deck_location> " : "",
    "</deck_location>" : "",
    "<deck_name> " : "",
    "</deck_name> " : "",
    "</deck_name>" : "",
    "<slide_name> " : "",
    "</slide_name> " : "",
    "</slide_name>" : "",
    "<slide_number> " : "",
    "</slide_number> " : "",
    "</slide_number>" : "",
    # using order_ref to support teams file recency
    "<teamcalendar_starttime> " : "",
    "</teamcalendar_starttime> " : "",
    "</teamcalendar_starttime>" : "",
    "<teammeeting_quantifier> " : "",
    "</teammeeting_quantifier>" : "",
    "<teammeeting_quantifier> " : "",
    "<teamspace_tab> " : "",
    "</teamspace_tab>" : "",
    "<teamspace_tab> " : "",
    "<teamspace_unclearkeyword> " : "",
    "</teamspace_unclearkeyword>" : "",
    "<teamspace_unclearkeyword> " : "",
    "<teamsuser_activitytype> " : "",
    "</teamsuser_activitytype>" : "",
    "<teamsuser_activitytype> " : "",
    "<teamsuser_status> " : "",
    "</teamsuser_status>" : "",
    "<teamsuser_status> " : "",
    "<teamsuser_topic> " : "",
    "</teamsuser_topic>" : "",
    "<teamsuser_topic> " : "",
    }
# planning to have from_contact_name and contact_name at the same time
# in this case, we will tage my my i I so no need this replacement
#removeSpecialSlotValue = {
#    "<contact_name> my </contact_name>":"my",
#    "<contact_name> My </contact_name>":"My",
#    "<contact_name> i </contact_name>":"i",
#    "<contact_name> I </contact_name>":"I",
#    }



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
    ' PowerPoint' : '<file_type> PowerPoint </file_type> ',
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
    # spec no longer being file_type
    #' spec' : '<file_type> spec </file_type> ',
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
    ' word' : '<file_type> word </file_type> ',


    # keep it as tag
    ' picture' : '<file_type> picture </file_type> ',
    ' music' : '<file_type> music </file_type> ',
    ' txt' : '<file_type> txt </file_type> ',
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
    '.PowerPoint' : '.<file_type> PowerPoint </file_type> ',
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

    # spec no longer being file_type    
    #'.spec' : '.<file_type> spec </file_type> ',
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
    '.word' : '.<file_type> word </file_type> ',

    # keep it as tag
    '.picture' : '.<file_type> picture </file_type> ',
    '.music' : '.<file_type> music </file_type> ',
    '.txt' : '.<file_type> txt </file_type> ',
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
    '. PowerPoint' : '. <file_type> PowerPoint </file_type> ',
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

    # spec no longer being file_type
    #'. spec' : '. <file_type> spec </file_type> ',
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
    '. word' : '. <file_type> word </file_type> ',

    # keep it as tag
    '. picture': '. <file_type> picture </file_type> ',
    '. music': '. <file_type> music </file_type> ',
    '. txt': '. <file_type> txt </file_type> ',
}

fileTypeNotTag={
    #' Word',
    #' word',
    #' Words',
    #' words',
    #" word's",
    #" Word's",
    ' text',
    ' Text',
    ' word pad',
    ' wordpad',
    ' note pad',
    ' notepad',
    ' adobe acrobat',
    ' xl',
    # . as word
    ' dot',
    ' Dot',
    ' scanned',    
    }

# i reverse as tag
# might be changed in the future
fileTypeReservedTag={
    ' Word',
    ' word',
    ' Words',
    ' words',
    " word's",
    " Word's",
    ' Microsoft word',
    ' microsoft word',
    ' Microsoft office word',
    ' microsoft office word',
    ' word xml',
    ' Excel sheet',
    ' excel sheet',
    }


fileRecencyReformat = {
    "<file_recency> the last used </file_recency>" : "the <order_ref> last </order_ref> <file_action> used </file_action>",
    "<file_recency> I worked on </file_recency>" : "<contact_name> I </contact_name> <file_action> worked </file_action> on",
    "<file_recency> I added a model </file_recency>" : "<contact_name> I </contact_name> <file_action> added </file_action> a model",
    "<file_recency> I have recently worked on </file_recency>" : "<contact_name> I </contact_name> have <file_recency> recently </file_recency> <file_action> worked </file_action> on",
    "<file_recency> that was closed </file_recency>" : "that was <file_action> used </file_action>",
    "<file_recency> I was working on </file_recency>" : "<contact_name> I </contact_name> was <file_action> working </file_action> on",
    "<file_recency> I added </file_recency>" : "<contact_name> I </contact_name> <file_action> added </file_action>",
    "<file_recency> I was using </file_recency>" : "<contact_name> I </contact_name> was <file_action> using </file_action>",
    # ? this case is not passivie so no action
    "<file_recency> I compose </file_recency>": "<contact_name> I </contact_name> compose",
    "<file_recency> I to compose </file_recency>": "<contact_name> I </contact_name> to compose",
    "<file_recency> I was to compose </file_recency>": "<contact_name> I </contact_name> was to compose",
    "<file_recency> I was compose </file_recency>" : "<contact_name> I </contact_name> was compose",
    "<file_recency> I have recently opened </file_recency>" : "<contact_name> I </contact_name> have <file_recency> recently </file_recency> <file_action> opened </file_action>",
    "<file_recency> i was composing </file_recency>": "<contact_name> i </contact_name> was <file_action> composing </file_action>",
    "<file_recency> I was composing </file_recency>": "<contact_name> I </contact_name> was <file_action> composing </file_action>",
    "<file_recency> I've recently used </file_recency>" : "<contact_name> I </contact_name> 've <file_recency> recently </file_recency> <file_action> used </file_action>",
    "<file_recency> I had been working on </file_recency>" : "<contact_name> I </contact_name> have been <file_action> working </file_action> on",
    "<file_recency> I recently morning </file_recency>" : "<contact_name> I </contact_name><file_recency> recently </file_recency> <file_action> morning </file_action>",
    "<file_recency> I was just working on </file_recency>" : "<contact_name> I </contact_name> was <file_recency> just </file_recency> <file_action> working </file_action> on", 
    "<file_recency> most recent </file_recency>" : "most <file_recency> recent </file_recency>",
    # makr to orderref
    "<file_recency> newest </file_recency>" : "<order_ref> newest </order_ref>",
    "<file_recency> latest </file_recency>" : "<order_ref> latest </order_ref>",
    # ? need to think if having last active as orderref
    "<file_recency> last active </file_recency>" : "<order_ref> last </order_ref> active",
    "<file_recency> I was creating </file_recency>" : "<contact_name> I </contact_name> was <file_action> creating </file_action>",
    "<file_recency> I just walked on </file_recency>" : "<contact_name> I </contact_name> <file_recency> just </file_recency> <file_action> walked </file_action> on",
    "<file_recency> I just edited </file_recency>" : "<contact_name> I </contact_name> <file_recency> just </file_recency> <file_action> edited </file_action>",
    "<file_recency> I was last working on </file_recency>" : "<contact_name> I </contact_name> was <order_ref> last </order_ref> <file_action> working </file_action> on",
    "<file_recency> I was recently working on </file_recency>" : "<contact_name> I </contact_name> was <file_recency> recently </file_recency> <file_action> working </file_action> on", 
    "<file_recency> added </file_recency>" : "<file_action> added </file_action>",
    "<file_recency> I had up before </file_recency>" : "<contact_name> I </contact_name> had <file_action> up </file_action> before",
    "<file_recency> I worked with Elizabeth on </file_recency>" : "<contact_name> I </contact_name> <file_action> worked </file_action> with <to_contact_name> Elizabeth </to_contact_name> on",
    "<file_recency> I uploaded </file_recency>" : "<contact_name> I </contact_name> <file_action> uploaed </file_action>",
    "<file_recency> I updated </file_recency>" : "<contact_name> I </contact_name> <file_action> updated </file_action>",
    "<file_recency> I last edited </file_recency>" : "<contact_name> I </contact_name> <order_ref> last </order_ref> <file_action> edited </file_action>",
    "<file_recency> I was working on recently </file_recency>" : "<contact_name> I </contact_name> was <file_action> working </file_action> on <file_recency> recently </file_recency>",
    "<file_recency> I just walked down </file_recency>" : "<contact_name> I </contact_name> <file_recency> just </file_recency> <file_action> walked </file_action> down",
    "<file_recency> i was working on </file_recency>" : "<contact_name> i </contact_name> was <file_action> working </file_action> on",
    "<file_recency> I shared with <to_contact_name> me </to_contact_name>" : "<contact_name> I </contact_name> <file_action> shared </file_action> with <to_contact_name> me </to_contact_name>",
    "<file_recency> I was editing </file_recency>" : "<contact_name> I </contact_name> was <file_action> editing </file_action>",
    "<file_recency> I was working on last </file_recency>" : "<contact_name> i </contact_name> was <file_action> working </file_action> on <order_ref> last </order_ref>",
    "<file_recency> I used most recently </file_recency>" : "<contact_name> I </contact_name> <file_action> used </file_action> most <file_recency> recent </file_recency>",
    "<file_recency> I was working </file_recency>" : "<contact_name> I </contact_name> was <file_action> working </file_action>",
    "<file_recency> I was writing </file_recency>" : "<contact_name> I </contact_name> was <file_action> writing </file_action>",
    "<file_recency> recently I was working on </file_recency>" : "<file_recency> recently </file_recency> <contact_name> I </contact_name> was <file_action> working </file_action> on",
    # not sure if new should be tag, do not new right now
    "<file_recency> I shared with me new </file_recency>" : "<contact_name> I </contact_name> <file_action> shared </file_action> with <to_contact_name> me </to_contact_name> new",
}
'''
fileTypeReformat={
    # download the powerpoint / powerpoints, qp
    <file_type> powerpoint </file_type>
    <file_type> power point </file_type>
    <file_type> powerpoints </file_type>
    <file_type> PowerPoint </file_type>
    
    # document / documents
    # 08092019, currently not tagged but being requsted for bug in teams
    <file_type> documents </file_type>
    #Open up my collision memo, qp
    <file_type> memo </file_type>
    <file_type> memos </file_type>
    #
Share voice skills pictures I was last working on with the meeting, qp
    # picture , pictures
    <file_type> pictures </file_type>
    # Open up my window cleaners jpg
    <file_type> jpg </file_type>

    # confirm but weird
    <file_type> excel </file_type>
    <file_type> spreadsheet </file_type>
    <file_type> spread sheet </file_type>
    <file_type> excel spreadsheet </file_type>

    
    # obvious no need to try
    <file_type> PPTX </file_type> 
    <file_type> powerpoint deck </file_type>
    <file_type> onenote </file_type>
    <file_type> ppt </file_type>
    <file_type> Word </file_type>
    <file_type> deck PowerPoint </file_type>
    <file_type> deck template </file_type>
    <file_type> OneNote </file_type>
    <file_type> one note </file_type>
    <file_type> csv </file_type>
    <file_type> pdf </file_type>
    <file_type> PowerPoint deck </file_type>
    <file_type> deck </file_type>
    <file_type> power point deck </file_type>

    # no trigger in qp
    <file_type> gif </file_type>
    <file_type> slide </file_type>
    <file_type> txt </file_type>
    <file_type> jpeg </file_type>
    <file_type> note </file_type>
    <file_type> vso </file_type>
    

    # download should be file_type to support teamspace_naviation
    <file_type> download </file_type>
    <file_type> downloads </file_type>

    # not in teams
    # ? wait for further discussion
    <file_type> list </file_type>
    <file_type> word document </file_type>
    <file_type> word doc </file_type>
}
'''

blackListQuerySet = {
    }

##############################
# intent level candidate
##############################
teamspaceSearchCandidateSet = set()

##############################
# slot level candidate
##############################



fileKeywordCandidateSet = set()
fileNameCandidateSet = set()
meetingStarttimeCandidateSet = set()
fileTypeCandidateSet = set()
dateCandidateSet = set()
timeCandidateSet = set()

# this is for deduplication and replacement
fileRecencyCandidateSet = set()

sharetargetTypeCandidateSet = set()
sharetargetNameCandidateSet = set()
contactNameCandidateSet = set()

fileActionCandidateSet = set()


orderRefCandidateSet = set()

# deduplication
skipQueryCandidateSet = set()


Output = [];


inputFile = 'files_slot_training.tsv'
with codecs.open(inputFile, 'r', 'utf-8') as fin:
    for line in fin:
        line = line.strip();
        if not line:
            continue;
        linestrs = line.split("\t");
        # make sure it at least has
        # Query	ExternalFeature	Weight	Intent	Domain	Slot
        if len(linestrs) < 5:
            continue;

        #add new here
        slot = linestrs[4]
        # for extra target slot you want
        xmlpairs = re.findall("(<.*?>.*?<\/.*?>)", slot)

        for xmlpair in xmlpairs:

            # extra type and value for xml tag
            xmlTypeEndInd = xmlpair.find(">")

            xmlType = xmlpair[1:xmlTypeEndInd]

            xmlValue = xmlpair.replace("<"+xmlType+">", "")
            xmlValue = xmlValue.replace("</"+xmlType+">", "")
            xmlValue = xmlValue.strip()

            if xmlType.lower() == "date":
                dateCandidateSet.add(xmlValue)
            if xmlType.lower() == "time":
                timeCandidateSet.add(xmlValue)   


"""
# comment shuffle in the first place
#random.shuffle(OutputSet);
"""


#######################
# slot level output
#######################

with codecs.open((inputFile.split("."))[0] +'_after_filtering_date.tsv', 'w', 'utf-8') as fout:
    for item in dateCandidateSet:
        fout.write(item + '\r\n');

with codecs.open((inputFile.split("."))[0] +'_after_filtering_time.tsv', 'w', 'utf-8') as fout:
    for item in timeCandidateSet:
        fout.write(item + '\r\n');





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

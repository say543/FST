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



fileDomainRelatedIntent = ['file_search', 'file_open', 'file_share', 'file_download', 'file_other', 'file_navigate', "teamspace_search"]

fileDomainPreviousRelatedIntent = ['file_search', 'file_open', 'file_share', 'file_download', 'file_other', 'file_navigate', "teamspace_search"]


teamsDomainToFileDomain = {
    "teams" : "files",
    "TEAMS" : "FILES"
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
    "</teamspace_channel >" : "</sharetarget_name>",
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


filesSlotRemoveSpace = {
    "<file_name> ": "<file_name>",
    " </file_name>": "</file_name>",
    "<file_type> ": "<file_type>",
    " </file_type>": "</file_type>",
    "<data_source> ": "<data_source>",
    " </data_source>": "</data_source>",
    "<contact_name> ": "<contact_name>",
    " </contact_name>": "</contact_name>",
    "<to_contact_name> ": "<to_contact_name>",
    " </to_contact_name>": "</to_contact_name>",
    "<file_keyword> ": "<file_keyword>",
    " </file_keyword>": "</file_keyword>",
    "<date> ": "<date>",
    " </date>": "</date>",
    "<time> ": "<time>",
    " </time>": "</time>",
    "<meeting_starttime> ": "<meeting_starttime>",
    " </meeting_starttime>": "</meeting_starttime>",
    "<file_action> ": "<file_action>",
    " </file_action>": "</file_action>",
    "<file_action_context> ": "<file_action_context>",
    " </file_action_context>": "</file_action_context>",
    "<position_ref> ": "<position_ref>",
    " </position_ref>": "</position_ref>",
    "<order_ref> ": "<order_ref>",
    " </order_ref>": "</order_ref>",
    "<file_recency> ": "<file_recency>",
    " </file_recency>": "</file_recency>",
    "<sharetarget_type> ": "<sharetarget_type>",
    " </sharetarget_type>": "</sharetarget_type>",
    "<sharetarget_name>": "<sharetarget_name>",
    " </sharetarget_name>": "</sharetarget_name>",
    "<file_folder> ": "<file_folder>",
    " </file_folder>": "</file_folder>",
}

# in the future read from file might be better
'''
fileRecencyCano ={
    "recent":"recent",
    "Recent":"recent",
    "recently":"recently",
    "Recently":"Recently",
    "just":"just",
    "Just": "Just"
    }
fileTypeCano ={
    "deck":"ppt",
    #downloaded file	download
    #downloaded files	download
    "downloaded":"download",
    "downloads":"download",
    "jpeg": "jpg",
    "memos": "memo",
    "one note" :"onenote",
    "pictures" :"picture",
    "power point": "ppt",
    "powerpoint":   "ppt",
    "spread sheet": "excel",
    "spreadsheet":  "excel",
    "text":	"txt",
    "word document":    "word",
    "word doc":	"word",
    "word docs"	"word",
    }

positionRefCano ={}
sharetargetType ={}
'''



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
# canonical merge
##############################
# dictionary of dictionary
# with sub dictionary for each canonical value
cano={
    }

#canofiles = glob.glob(r"../canonical_collect/*.txt");
canofiles = ["file_recency.txt", "file_type.txt", "order_ref.txt", "position_ref.txt", "sharetarget_type.txt"];
for file in canofiles:
    filestr = os.path.basename(file)
    filestr = filestr.split('.');
    #print(filestr[0])

    key =filestr[0]
    cano[key] = {}

    

    print("collecting: " + file + "for" + key );
    with codecs.open(file, 'r', 'utf-8') as fin:
        for line in fin:
            line = line.strip();
            if not line:
                continue;
            array = line.split('\t');
            if len(array) < 2:
                print("error:" + line);

            cano[key][array[0]] = array[1]
    

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

# this is for deduplication and replacement
fileRecencyCandidateSet = set()

sharetargetTypeCandidateSet = set()
sharetargetNameCandidateSet = set()
contactNameCandidateSet = set()

fileActionCandidateSet = set()


orderRefCandidateSet = set()


taskFrameDialogEntities = set()
taskFrameEntityStates = set()

# deduplication
skipQueryCandidateSet = set()


Output = [];
OutputSlotEvaluation = [];

OutputIntentEvaluation = [];


with codecs.open('Teams-golden.tsv', 'r', 'utf-8') as fin:
    for line in fin:
        line = line.strip();
        if not line:
            continue;
        linestrs = line.split("\t");

        # make sure it at least has
        # Query	ExternalFeature	Weight	Intent	Domain	Slot
        if len(linestrs) < 11:
            continue;



        # processing conversation context for multi turn
        # here file_title alawys go to file_keyword
        # update it later


        conversationContext = '{}'
        if len(linestrs[9]) > 0 and linestrs[9] != "ConversationContext":        


            for key in sorted (teamsSlotToFileSlot.keys()):
                if not key.startswith(" </") and not key.startswith("</"):
                    value = teamsSlotToFileSlot[key]
                    valueWoParenthese = (value.strip())[1:len(value.strip())-1]
                    keyWoParenthese = (key.strip())[1:len(key.strip())-1]
                    linestrs[9] = linestrs[9].replace(keyWoParenthese, valueWoParenthese)

            if len(linestrs[9]) >0:
                #print(linestrs[9])
                conversationContext = json.loads(linestrs[9])

                

            # map teams domain to files domain
            #if 'PreviousTurnDomain' in conversationContext:
                #print(conversationContext['PreviousTurnDomain'])
                #conversationContext['PreviousTurnDomain'] = teamsDomainToFileDomain[conversationContext['PreviousTurnDomain'][0]]

            #if 'TaskFrameDialogEntities' in  conversationContext:
            #    conversationContext['PreviousTurnDomain']

        



        # skip multi turn query
        if linestrs[0].startswith("Teams-Multiturn"):
            #conversationContext = json.loads(linestrs[9])
            #print(conversationContext)

            #print(conversationContext['TaskFrameDialogEntities'])



            if 'TaskFrameDialogEntities' in  conversationContext:
                taskFrameDialogEntities.add('\t'.join(conversationContext['TaskFrameDialogEntities']))

            if 'TaskFrameEntityStates' in  conversationContext:
                taskFrameEntityStates.add('\t'.join(conversationContext['TaskFrameEntityStates']))

            '''
            # if ont in valid
            
            if len(conversationContext['PreviousTurnIntent']) >0 and conversationContext['PreviousTurnIntent'][0] not in fileDomainPreviousRelatedIntent:
                continue
            if len(conversationContext['TaskFrameDialogEntities']) >0 and \
                len((conversationContext['TaskFrameDialogEntities'][0].split(":"))) >=2 and \ 
                (conversationContext['TaskFrameDialogEntities'][0].split(":"))[0] not in fileDomainPreviousRelatedIntent:
                continue
            if len(conversationContext['TaskFrameEntityStates']) >0 and \
                len((conversationContext['TaskFrameEntityStates'][0].split(":"))) >=2 and \ 
                (conversationContext['TaskFrameEntityStates'][0].split(":"))[0] not in fileDomainPreviousRelatedIntent:
                continue
            '''

            continue



        # make sure it is find_my_stuff intent
        if linestrs[6] in fileDomainRelatedIntent:

            # document teamspace_search
            # for further analysis
            # skip it at first
            if linestrs[6] == "teamspace_search":
                # having problem or tab
                # then skip
                if linestrs[4].find("program") != -1 or \
                   linestrs[4].find("tab") != -1 or \
                   linestrs[4].find("channel") != -1 or \
                   linestrs[4].find("team") != -1 or \
                   linestrs[4].find("teams") != -1 or \
                   linestrs[4].find("conversation") != -1 or \
                   linestrs[4].find("chat") != -1:
                    teamspaceSearchCandidateSet.add(line)
                    continue
                linestrs[6] = "file_search"

            slot = linestrs[7]
            for key in sorted (teamsSlotToFileSlot.keys()) :
                slot = slot.replace(key, teamsSlotToFileSlot[key])

            # planning to have from_contact_name and contact_name at the same time
            # in this case, we will tage my my i I so no need this replacement
            #for key in sorted (removeSpecialSlotValue.keys()) :
            #    slot = slot.replace(key, removeSpecialSlotValue[key])


            # recency re format
            for key in sorted (fileRecencyReformat.keys()) :
                slot = slot.replace(key, fileRecencyReformat[key])
            
                
            # remove head and end spaces 
            slot = slot.strip()


            # verb my, make my tag
            # only detect at the neginning of queries
            # with space as here for seperrator
            # verb my, make my tag
            # only detect at the neginning of queries
            # with space as here for seperrator
            prefixWtihVerbWithMy =set(["mystuff ",
                                       "Can you ",
                                       "can you ",
                                       "Please ",
                                       "please ",
                                       "Can you please ",
                                       "can you please ",
                                       "hey cortana please ",
                                       "Hey cortana please ",
                                       "cortana please ",
                                       "cortana ",
                                       "Hey cortana ",
                                       "hey cortana ",
                                       "Hey cortana can you ",
                                       "hey cortana can you ",
                                       "i want to ",
                                       "I want to ",
                                       "i need to ",
                                       "I need to ",
                                       "i have to ",
                                       "I have to ",
                                       "i want you to ",
                                       "I want you to ",
                                       "i can't ",
                                       "I can't ",
                                       "Try to ",
                                       "try to ",
                                       "Cortana, ",
                                       "cortana, ",
                                       "Cortana, please ",
                                       "cortana, please ",
                                       "Cortana, help me ",
                                       "cortana, help me ",
                                       "cortana, i need to ",
                                       "Cortana, i need to ",
                                       "cortana can you ",
                                       "Cortana can you "
                                       "hi what's up ",
                                       "Hi what's up ",
                                       "hello can you ",
                                       "Hello can you ",
                                       "can you help me ",
                                       "Can you help me ",
                                       "can you help ",
                                       "Can you help ",
                                       "help me ",
                                       "Help me ",
                                       "where do i ",
                                       "Where do i ",
                                       "where do I ",
                                       "Where do I ",
                                       "where will i ",
                                       "Where will i ",
                                       "where will I ",
                                       "Where will I ",
                                       "where can i ",
                                       "Where can i ",
                                       "where can I ",
                                       "Where can I ",
                                       "where do i ",
                                       "Where do i ",
                                       "where do I ",
                                       "Where do I ",
                                       "Siri ",
                                       "siri ",
                                       "Bing ",
                                       "bing ",
                                       ])
            # with space as here for seperrator
            verbsWithMy = set(["open ",
                               "find ",
                               "show ",
                               "Open ",
                               "Find ",
                               "Show ",
                               "search ",
                               "Search ",
                                ])
            mys = set(["my",
                      "My",
                      ])
            for prefix in prefixWtihVerbWithMy:
                for verb in verbsWithMy:
                    for my in mys:
                        if slot.startswith(prefix + verb + my):
                            slot = slot.replace(prefix + verb + my, prefix + verb + "<contact_name> " + my +" </contact_name>")
                        elif slot.startswith(verb + my):
                            slot = slot.replace(verb + my, verb + "<contact_name> " + my +" </contact_name>")

            # fine-grained parse
            #list = re.findall("(</?[^>]*>)", slot)


            # missed date from teams
            dates = set(["yesterday",
                         "last night",
                         "today"
                        ]
                       )
            for date in dates:
                # if found token but not being taked
                if slot.find(date) != -1 and slot.find("<date> "+ date) == -1 and slot.find("<meeting_starttime> "+ date) == -1:
                    slot = slot.replace(date, "<date> "+ date +" </date>")
                # if found token but not being taked (no end tag detection since last)
                if slot.find(date) != -1 and slot.find("<date> "+ date) == -1 and slot.find("<meeting_starttime> "+ date) == -1:
                    slot = slot.replace(date, "<date> "+ date +" </date>")


            # missed data source
            datasources = set(["onedrive",
                        ]
                       )
            for datasource in datasources:
                # if found token but not being taked
                if slot.find(datasource) != -1 and slot.find("<data_source> "+ datasource) == -1:
                    slot = slot.replace(datasource, "<data_source> "+ datasource +" </data_source>")

            # missed time
            times = set(["this morning",
                        ]
                        )
            for time in times:
                # if found token but not being taked
                if slot.find(time) != -1 and slot.find("<time> "+ time) == -1 and slot.find("<meeting_starttime> "+ time) == -1:
                    slot = slot.replace(time, "<time> "+ time +" </time>")
                # if found token but not being taked (no end tag detection since last)
                if slot.find(time) != -1 and slot.find("<time> "+ time) == -1 and slot.find("<meeting_starttime> "+ time) == -1:
                    slot = slot.replace(time, "<time> "+ time +" </time>")


            if slot.find("<time> recent </time>") != -1:
                slot = slot.replace("<time> recent </time>", "<file_recency> recent </file_recency>")

            # handle my
            # to my
            
            if slot.find(" to my ") != -1:
                slot = slot.replace(" to my", " to <contact_name> my </contact_name>")
            # with me
            if slot.find(" to me ") != -1:
                slot = slot.replace(" to me", " to <to_contact_name> me </to_contact_name>")
            if slot.find(" with me ") != -1:
                slot = slot.replace(" with me ", " with <to_contact_name> me </to_contact_name>")

                
            
            # i verb
            verbsAlongWithContactName = set(["downloaded",
                     "worked",
                     "created",
                     "saved",
                     "made",
                     "edited",
                     "took",
                     "uploaded",
                     "working",
                     "shared",
                     "wrote",
                     "added",
                     "used",
                     "using",
                     "composed",
                     "opened",
                     "composing",
                     "morning",
                     "walked",
                     "edited",
                     "updated",
                     "writing",
                     "doing",
                     "did",
                     "looking",
                     "looked",
                     "reviewed",
                     ])
            contactNames = ["i",
                            "I"
                            ]
            for verb in verbsAlongWithContactName:
                for contactName in contactNames:
                    # with file action already
                    # try to tag contact name
                    ## will suffer big / small case problems but leave it there eg: query small but annotation cbig
                    if linestrs[0].find(contactName +" "+ verb) != -1 and slot.find(contactName +" <file_action> "+verb+" </file_action>")!=-1:
                        slot = slot.replace(contactName +" <file_action> "+verb+" </file_action>", "<contact_name> "+contactName +" </contact_name>"+ " <file_action> "+verb+" </file_action>")
                    if linestrs[0].find(contactName +" was "+ verb) != -1 and slot.find(contactName +" was <file_action> "+verb+" </file_action>")!=-1:
                        slot = slot.replace(contactName +" was <file_action> "+verb+" </file_action>", "<contact_name> "+contactName +" </contact_name>"+ " was <file_action> "+verb+" </file_action>")

            
                #if linestrs[0].find("i downloaded") != -1 and slot.find("i <file_action> downloaded </file_action>")!=-1:
                #    slot = slot.replace("i <file_action> downloaded </file_action>", "<contact_name> i </contact_name> <file_action> downloaded </file_action>")
            
                #if linestrs[0].find("I downloaded") != -1 and slot.find("I <file_action> downloaded </file_action>")!=-1:
                #    slot = slot.replace("i <file_action> downloaded </file_action>", "<contact_name> I </contact_name> <file_action> downloaded </file_action>")


            # tailor downloads special cases for teams
            # add new slot file_folder  for downloads
            if slot.find("<file_type> downloads </file_type>") != -1:
                slot = slot.replace("<file_type> downloads </file_type>", "<file_folder> downloads </file_folder>")
            if slot.find("<file_type> download </file_type>") != -1:
                slot = slot.replace("<file_type> download </file_type>", "<file_folder> download </file_folder>")

            '''
            # hanlde my xxx and my should be contact name
            # ? not sure why this sniffet does not work
            ##nouns = [
            ##    "party"
            ##    ]
            ##for noun in nouns:
            ##    if slot.find("<keyword> my "+noun+" </keyword>")!= -1:
            ##        slot = slot.replace("<keyword> my " + noun +" </keyword>", "<contact_name> my </contact_name>"+" <keyword> "+ noun +" </keyword>")
    
            # tailor downloads special cases for teams
            if slot.find("<data_source> downloads </data_source>") != -1:
                slot = slot.replace("<data_source> downloads </data_source>", "<file_type> downloads </file_type>")

            # start time too long handle
            if slot.find("<time> in the last hour </time>") != -1:zc
                slot = slot.replace("<time> in the last hour </time>", "in the <time> last hour </time>")
            if slot.find("<time> within the last hour </time>") != -1:
                slot = slot.replace("<time> within the last hour </time>", "within the <time> last hour </time>")

            if slot.find("<time> in the last day </time>") != -1:
                slot = slot.replace("<time> in the last day </time>", "in the <time> last day </time>")
            if slot.find("<time> within the last day </time>") != -1:
                slot = slot.replace("<time> within the last day </time>", "within the <time> last day </time>")


            if slot.find("<date> in the last month </date>") != -1:
                slot = slot.replace("<date> in the last month </date>", "in the <date> last month </date>")
            if slot.find("<date> within the last 1 month </date>") != -1:
                slot = slot.replace("<date> within the last month </date>", "within the <date> last month </date>")


            if slot.find("<date> in the last week </date>") != -1:
                slot = slot.replace("<date> in the last week </date>", "in the <date> last week </date>")
            if slot.find("<date> within the last week </date>") != -1:
                slot = slot.replace("<date> within the last 1 week </date>", "within the <date> last week </date>")


            #https://blog.csdn.net/blueheart20/article/details/52883045
            for num in range(1, 400):
                if slot.find("<date> in the last " + str(num) + " hours </date>") != -1:
                    slot = slot.replace("<date> in the last " + str(num) + " hours </date>", "in the <date> last " + str(num) + " hours </date>")
                if slot.find("<date> within the last " + str(num) + " hours </date>") != -1:
                    slot = slot.replace("<date> within the last " + str(num) + " hours </date>", "within the <date> last " + str(num) + " hours </date>")

                if slot.find("<date> in the last " + str(num) + " days </date>") != -1:
                    slot = slot.replace("<date> in the last " + str(num) + " days </date>", "in the <date> last " + str(num) + " days </date>")
                if slot.find("<date> within the last " + str(num) + " days </date>") != -1:
                    slot = slot.replace("<date> within the last " + str(num) + " days </date>", "within the <date> last " + str(num) + " days </date>")

                
                if slot.find("<date> in the last " + str(num) + " months </date>") != -1:
                    slot = slot.replace("<date> in the last " + str(num) + " months </date>", "in the <date> last " + str(num) + " months </date>")
                if slot.find("<date> within the last " + str(num) + " months </date>") != -1:
                    slot = slot.replace("<date> within the last " + str(num) + " months </date>", "within the <date> last " + str(num) + " months </date>")

                if slot.find("<date> in the last " + str(num) + " weeks </date>") != -1:
                    slot = slot.replace("<date> in the last " + str(num) + " weeks </date>", "in the <date> last " + str(num) + " weeks </date>")
                if slot.find("<date> within the last " + str(num) + " weeks </date>") != -1:
                    slot = slot.replace("<date> within the last " + str(num) + " weeks </date>", "within the <date> last " + str(num) + " weeks </date>")
                    
            alphaDigits = ["two",
                          "three",
                          "four",
                          "five",
                          "six",
                          "seven",
                          "eight",
                          "night",
                          "ten",
                          "eleven",
                          "twelve"
                          ]
            for alphadigit in alphaDigits:
                if slot.find("<date> in the last " + alphadigit + " hours </date>") != -1:
                    slot = slot.replace("<date> in the last " + alphadigit + " hours </date>", "in the <date> last " + alphadigit + " hours </date>")
                if slot.find("<date> within the last " + alphadigit + " hours </date>") != -1:
                    slot = slot.replace("<date> within the last " + alphadigit + " hours </date>", "within the <date> last " + alphadigit + " hours </date>")

                if slot.find("<date> in the last " + alphadigit + " days </date>") != -1:
                    slot = slot.replace("<date> in the last " + alphadigit + " days </date>", "in the <date> last " + alphadigit + " days </date>")
                if slot.find("<date> within the last " + alphadigit + " days </date>") != -1:
                    slot = slot.replace("<date> within the last " + alphadigit + " days </date>", "within the <date> last " + alphadigit + " days </date>")
                    
                if slot.find("<date> in the last " + alphadigit + " months </date>") != -1:
                    slot = slot.replace("<date> in the last " + alphadigit + " months </date>", "in the <date> last " + alphadigit + " months </date>")
                if slot.find("<date> within the last " + alphadigit + " months </date>") != -1:
                    slot = slot.replace("<date> within the last " + alphadigit + " months </date>", "within the <date> last " + alphadigit + " months </date>")

                if slot.find("<date> in the last " + alphadigit + " weeks </date>") != -1:
                    slot = slot.replace("<date> in the last " + alphadigit + " weeks </date>", "in the <date> last " + alphadigit + " weeks </date>")
                if slot.find("<date> within the last " + alphadigit + " weeks </date>") != -1:
                    slot = slot.replace("<date> within the last " + alphadigit + " weeks </date>", "within the <date> last " + alphadigit + " weeks </date>")


            if slot.find("<from_contact_name>") != -1 and slot.find("<contact_name>") != -1:
                # do contact_name replacement then do from_contact_name replacement inorder
                slot = slot.replace("<contact_name>", "<to_contact_name>")
                slot = slot.replace("</contact_name>", "</to_contact_name>")
                slot = slot.replace("<from_contact_name>", "<contact_name>")
                slot = slot.replace("</from_contact_name>", "</contact_name>")
            elif slot.find("<from_contact_name>") != -1:
                slot = slot.replace("<from_contact_name>", "<contact_name>")
                slot = slot.replace("</from_contact_name>", "</contact_name>")

            #for recent, recently, just . Just in start_time, mapping to order_ref
            if slot.find("<time> recent </time>") != -1:
                slot = slot.replace("<time> recent </time>", "<file_recency> recent </file_recency>")
            if slot.find("<time> Recent </time>") != -1:
                slot = slot.replace("<time> Recent </time>", "<file_recency> Recent </file_recency>")
            if slot.find("<time> recently </time>") != -1:
                slot = slot.replace("<time> recently </time>", "<file_recency> recently </file_recency>")
            if slot.find("<time> Recently </time>") != -1:
                slot = slot.replace("<time> Recently </time>", "<file_recency> Recently </file_recency>")
            if slot.find("<time> just </time>") != -1:
                slot = slot.replace("<time> just </time>", "<file_recency> just </file_recency>")
            if slot.find("<time> Just </time>") != -1:
                slot = slot.replace("<time> Just </time>", "<file_recency> Just </file_recency>")
            if slot.find("<order_ref> most recent </order_ref>") != -1:
                slot = slot.replace("<order_ref> most recent </order_ref>", "most <file_recency> recent </file_recency>")
            if slot.find("<order_ref> recent </order_ref>") != -1:
                slot = slot.replace("<order_ref> recent </order_ref>", "<file_recency> recent </file_recency>")
            if slot.find("<order_ref> Recent </order_ref>") != -1:
                slot = slot.replace("<order_ref> Recent </order_ref>", "<file_recency> Recent </file_recency>")
            '''

            # for contact_name to reanme to to_contact_name
            xmlpairs = re.findall("(<.*?>.*?<\/.*?>)", slot)

            for xmlpair in xmlpairs:
                # file_keywrod to file_name
                #https://stackoverflow.com/questions/41484526/regular-expression-for-matching-non-whitespace-in-python
                # not perfect but good enough
                if xmlpair.startswith("<file_keyword>") and re.search(r'[\S]+\.[\S]+', xmlpair) is not None:
                    newPair = xmlpair.replace("<file_keyword>", "<file_name>")
                    newPair = newPair.replace("</file_keyword>", "</file_name>")
                    slot = slot.replace(xmlpair, newPair)

                # to
                # with
                # from
                if xmlpair.startswith("<contact_name>") and slot.find("with <contact_name>") != -1 and slot.find("<file_action>") != -1 and slot.find("<file_action>") < slot.find(xmlpair):
                    #print("inside")
                    newPair = xmlpair.replace("<contact_name>", "<to_contact_name>")
                    newPair = newPair.replace("</contact_name>", "</to_contact_name>")
                    slot = slot.replace(xmlpair, newPair)

                # to deal with 'go to my fils' / hey could you please direct me to my last files that was used ?
                # make sure it has file_action slot and it happens before
                if xmlpair.startswith("<contact_name>") and slot.find("to <contact_name>") != -1  and slot.find("<file_action>") != -1 and slot.find("<file_action>") < slot.find(xmlpair):
                    newPair = xmlpair.replace("<contact_name>", "<to_contact_name>")
                    newPair = newPair.replace("</contact_name>", "</to_contact_name>")
                    slot = slot.replace(xmlpair, newPair)
                if xmlpair.startswith("<contact_name>") and slot.find("from <contact_name>") != -1 and slot.find("<file_action>") != -1 and slot.find("<file_action>") < slot.find(xmlpair):
                    newPair = xmlpair.replace("<contact_name>", "<to_contact_name>")
                    newPair = newPair.replace("</contact_name>", "</to_contact_name>")
                    slot = slot.replace(xmlpair, newPair)


                # to deal with "file to <contact_name> xxx </contact_name>"
                if xmlpair.startswith("<contact_name>") and slot.find("file to <contact_name>") != -1 and slot.find("file to <contact_name>") < slot.find(xmlpair):
                    newPair = xmlpair.replace("<contact_name>", "<to_contact_name>")
                    newPair = newPair.replace("</contact_name>", "</to_contact_name>")
                    slot = slot.replace(xmlpair, newPair)

                if xmlpair.startswith("<contact_name>") and slot.find("with <contact_name>") != -1 and slot.find("with <contact_name>") < slot.find(xmlpair):
                    newPair = xmlpair.replace("<contact_name>", "<to_contact_name>")
                    newPair = newPair.replace("</contact_name>", "</to_contact_name>")
                    slot = slot.replace(xmlpair, newPair)

            # after all slot replacement
            # remove head and end spaces 
            slot = slot.strip()


            metadata = ""

            # for analysis
            xmlpairs = re.findall("(<.*?>.*?<\/.*?>)", slot)           
            #print (xmlpairs)
            for xmlpair in xmlpairs:

                xmlTypeEndInd = xmlpair.find(">")

                xmlType = xmlpair[1:xmlTypeEndInd]

                xmlValue = xmlpair.replace("<"+xmlType+">", "")
                xmlValue = xmlValue.replace("</"+xmlType+">", "")
                xmlValue = xmlValue.strip()

                #print(xmlType)
                #print(xmlValue)
                
                if xmlpair.startswith("<file_keyword>"):
                    fileKeywordCandidateSet.add(xmlpair)
                if xmlpair.startswith("<file_name>"):
                    fileNameCandidateSet.add(xmlpair)
                if xmlpair.startswith("<meeting_starttime>"):
                    meetingStarttimeCandidateSet.add(xmlpair)
                if xmlpair.startswith("<file_type>"):
                    fileTypeCandidateSet.add(xmlpair)
                    if xmlType  in cano and xmlValue in cano[xmlType]:
                        metadata = metadata + ',' +'{"slot_name":"' + xmlType + '","slot_value":"'+xmlValue+'","meta_data":{"CanonicalEntity":"'+ cano[xmlType][xmlValue]+'"}}'
                if xmlpair.startswith("<file_recency>"):
                    # this is for replacement profiling
                    fileRecencyCandidateSet.add(xmlpair)
                    
                    if xmlType  in cano and xmlValue in cano[xmlType]:
                        metadata = metadata + ',' +'{"slot_name":"' + xmlType + '","slot_value":"'+xmlValue+'","meta_data":{"CanonicalEntity":"'+ cano[xmlType][xmlValue]+'"}}'
             
                if xmlpair.startswith("<sharetarget_type>"):
                    sharetargetTypeCandidateSet.add(xmlpair)
                    if xmlType  in cano and xmlValue in cano[xmlType]:
                        metadata = metadata + ',' +'{"slot_name":"' + xmlType + '","slot_value":"'+xmlValue+'","meta_data":{"CanonicalEntity":"'+ cano[xmlType][xmlValue]+'"}}'

                if xmlpair.startswith("<sharetarget_name>"):
                    sharetargetNameCandidateSet.add(xmlpair)
                    if xmlType  in cano and xmlValue in cano[xmlType]:
                        metadata = metadata + ',' +'{"slot_name":"' + xmlType + '","slot_value":"'+xmlValue+'","meta_data":{"CanonicalEntity":"'+ cano[xmlType][xmlValue]+'"}}'
                if xmlpair.startswith("<contact_name>"):
                    contactNameCandidateSet.add(xmlpair)
                if xmlpair.startswith("<file_action>"):
                    fileActionCandidateSet.add(xmlpair)
                if xmlpair.startswith("<order_ref>"):
                    orderRefCandidateSet.add(xmlpair)
                    if xmlType  in cano and xmlValue in cano[xmlType]:
                        metadata = metadata + ',' +'{"slot_name":"' + xmlType + '","slot_value":"'+xmlValue+'","meta_data":{"CanonicalEntity":"'+ cano[xmlType][xmlValue]+'"}}'
                if xmlpair.startswith("<position_ref>"):
                    if xmlType  in cano and xmlValue in cano[xmlType]:
                        metadata = metadata + ',' +'{"slot_name":"' + xmlType + '","slot_value":"'+xmlValue+'","meta_data":{"CanonicalEntity":"'+ cano[xmlType][xmlValue]+'"}}'



            # remove head,
            if len(metadata) > 0:
                metadata = metadata[1:len(metadata)]
            metadata = "[" +metadata +"]"
    
            # id / message id / message time stamp / message from/ message text / judged domain / judge d intent / JudgedConstraint
            #Output.append(linestrs[0]+"\t"+linestrs[1]+"\t"+linestrs[2]+"\t"+linestrs[3]+"\t"+linestrs[4]+"\t"+teamsDomainToFileDomain[linestrs[5]]+"\t"+linestrs[6]+"\t"+slot);

            # original format
            # id / message id / message time stamp / message from/ message text / judged domain / judge d intent / JudgedConstraint / MetaData	/ConversationContext	/Frequency  /ImplicitConstraints
            # for evaluation need to replace space folloiwng XML tag
            slotRemoveSpaceAfterXML = slot
            for key in filesSlotRemoveSpace :
                slotRemoveSpaceAfterXML = slotRemoveSpaceAfterXML.replace(key, filesSlotRemoveSpace[key])
            Output.append(linestrs[0]+"\t"+linestrs[1]+"\t"+linestrs[2]+"\t"+linestrs[3]+"\t"+linestrs[4]+"\t"+teamsDomainToFileDomain[linestrs[5]]+"\t"+linestrs[6]+"\t"+slotRemoveSpaceAfterXML+"\t"+metadata+"\t"+linestrs[9]+"\t"+linestrs[10]+"\t"+"");
            
            #message text / judged domain / judge d intent / JudgedConstraint
            #Output.append(linestrs[3]+"\t"+linestrs[4]+"\t"+teamsDomainToFileDomain[linestrs[5]]+"\t"+linestrs[6]+"\t"+slot);

            # id / message / intent / domain / constraint
            # for training purpose's format
            
            OutputSlotEvaluation.append("0"+"\t"+linestrs[4]+"\t"+linestrs[6]+"\t" +teamsDomainToFileDomain[linestrs[5]]+"\t"+slot);

            # TurnNumber / PreviousTurnIntent / query /intent
            # for training purpose's format
            OutputIntentEvaluation.append("0"+"\t"+""+"\t"+linestrs[4]+"\t" +linestrs[6]);

"""
# comment shuffle in the first place
#random.shuffle(OutputSet);
"""

# for judge trainer format
with codecs.open('teams_golden_after_filtering.tsv', 'w', 'utf-8') as fout:

    # if outout originla format
    fout.write("ConversationId\tMessageId\tMessageTimestamp\tMessageFrom\tMessageText\tJudgedDomain\tJudgedIntent\tJudgedConstraints\tMetaData\tConversationContext\tFrequency\tImplicitConstraints\r\n")
    for item in Output:
        fout.write(item + '\r\n');

# for CMF slot evaluation format
with codecs.open('teams_golden_after_filtering_slot_evaluation.tsv', 'w', 'utf-8') as fout:

    # if output for traing
    fout.write("id\tquery\tintent\tdomain\tQueryXml\r\n")
    for item in OutputSlotEvaluation:
        fout.write(item + '\r\n');

# for CMF intent evaluation format
with codecs.open('teams_golden_after_filtering_intent_evaluation.tsv', 'w', 'utf-8') as fout:

    # if output for traing
    fout.write("TurnNumber\tPreviousTurnIntent\tquery\tintent\r\n")
    for item in OutputIntentEvaluation:
        fout.write(item + '\r\n');

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

with codecs.open('teams_slot_training_after_filtering_task_frame_dialog.tsv', 'w', 'utf-8') as fout:
    for item in taskFrameDialogEntities:
        fout.write(item + '\r\n');

with codecs.open('teams_slot_training_after_filtering_task_frame_entity.tsv', 'w', 'utf-8') as fout:
    for item in taskFrameEntityStates:
        fout.write(item + '\r\n');



#######################
# query replacement revert
#######################

import codecs;
import random;
import re;


import math
import re
import sys

# add hyper paramter if unbalanced
hyper_parameter = 200



#fileDomainRelatedIntent = ['file_search', 'file_open', 'file_share', 'file_download', 'file_other']

myStuffIntentToFileIntent = {
    "find_my_stuff" : "file_search"
}

myStuffDomainToFileDomain = {
    "mystuff" : "files"
}



# miss <action context> and it will need to check manually
myStuffSlotToFileSlot = {
    "<title>" : "<file_name>",
    "</title>" : "</file_name>",
    "<data_type>" : "<file_type>",
    "</data_type>" : "</file_type>",
    # rename file_location ti data source, so no need this mapping
    #"<data_source>" : "<file_location>",
    #"</data_source>" : "</file_location>",
    # planning to have from_contact_name and contact_name at the same time
    # no need to replace anymore
    #"<contact_name>" : "<contact_name>",
    #"</contact_name>" : "</contact_name>",
    #"<from_contact_name>" : "<contact_name>",
    #"</from_contact_name>" : "</contact_name>",    
    "<keyword>" : "<file_keyword>",
    "</keyword>" : "</file_keyword>",
	# only has date and time so map them to the same 
    "<start_date>" : "<date>",
    "</start_date>" : "</date>",
    "<start_time>" : "<time>",
    "</start_time>" : "</time>",
    "<end_date>" : "<date>",
    "</end_date>" : "</date>",
    "<end_time>" : "<time>",
    "</end_time>" : "</time>",
    #"<file_action>" : "<file_action>",
    #"</file_action>" : "</file_action>",
    #"<position_ref>" : "<position_ref>",
    #"</position_ref>" : "</position_ref>",
    # my stuff remove
    "<attachment> " : "",
    # based on sorting order, having space at the end will be checked at first
    # so check space version then no space
    "</attachment> " : "",
    "</attachment>" : "",
    "<data_destination> " : "",
    "</data_destination> " : "",
    "</data_destination>" : "",
    "<data_destination> " : "",
    "</data_destination> " : "",
    "</data_destination>" : "",
    "<location> " : "",
    "</location> " : "",
    "</location>" : "",
    # using order_ref to support teams file recency
    #"<order_ref> " : "",
    #"</order_ref> " : "",
    #"</order_ref>" : "",
    "<quantifier> " : "",
    "</quantifier>" : "",
    "<source_platform> " : "",
    "</source_platform> " : "",
    "</source_platform>" : "",
    "<transform_action> " : "",
    "</transform_action> " : "",
    "</transform_action>" : "",
    # one extra tag found by training result, remove as well
    "<mystuff_other> " : "",
    "</mystuff_other>" : "",
    }
# planning to have from_contact_name and contact_name at the same time
# in this case, we will tage my my i I so no need this replacement
#removeSpecialSlotValue = {
#    "<contact_name> my </contact_name>":"my",
#    "<contact_name> My </contact_name>":"My",
#    "<contact_name> i </contact_name>":"i",
#    "<contact_name> I </contact_name>":"I",
#    }



blackListQuerySet = {
    "go to my desktop",
    "internet explorer on this pc",
    "take me to documents",
    "go to my photo album",
    "i don't wanna go to the internet just tell me where it is on this computer",
    "search my computer for movie battles",
    "can you open my documents folder",
    "games on this phone",
    "can you show me my desktop",
    "show me my desktop icons",
    "c:\program files ( x86 ) \java\jre7\bin\javacpl.exe",
    "c:\\program files ( x86 ) \\java\\jre7\\bin\\javacpl.exe",
    "c:\program files(x86)\java\jre7\bin\javacpl. exe",
    "c:\\program files(x86)\\java\\jre7\\bin\\javacpl.exe",
    "c:program files java jre7 bin javacpl.exe",
    "downloads on this computer",
    "where is the microsoft word program app on this computer",
    "show me my desktop",
    "what videos do i have on my computer",
    "show photos on this pc",
    "open my computer",
    "open my folder",
    "where is wsa bbs2 dot exe file on the computer",
    "show me pictures in this computer",
    "can you open my picture folder",
    "find my music folder",
    "where can i find my favorites on my computer",
    "search my computer for discrete mathematics",
    "go to pictures",
    "go to my pictures",
    "cortana show me my games",
    "find microsoft word docs in my pc",
    "surf search for all p.d.f . 's on this device",
    "please show me my pictures",
    "show me photos of my car",
    "cortana show me my album",
    "find my recycle bin",
    "take me to my favorites",
    "take me to my documents",
    "go to my notes",
    "go to my files",
    "search my stuff for pictures",
    "to do list",
    "go to my saved items",
    "find downloads",
    "find download",
    "open downloads",
    "open download",
    "check my downloads",
    "check my download",
    "open my picture",
    "open my pictures",
    "show list",
    "show lists",
    "What's on my list?",
    "What's on my lists?",
    "tell me a list",
    "tell me lists",
    "find newest one under documents",
    "show me my music",
    "find my list of items to get at the grocery store",
    "open picture",
    "open pictures",
    "hi where are my photos",
    "find internet explorer",
    "let me see the list",
    "say hey cortana show me my task list",
    "show me some of my photos",
    "go to my xbox games",
    "show pictures",
    "show picture",
    "find me my music",
    "what are my what 's on my list today",
    "find a copy of my cd",
    "cortana where 's my documents",
    "can you show me my task list",
    "Where did I save my lasagna recipe from Tuesday?",
    "find pictures of my wife",
    "can you show me a picture of my old self ?",
    "old files",
    "On Tuesday, where did I save my Baked Ziti recipe?",
    "My Notes: Find Mom Notes",
    "can you show me a picture of my old self?",
    "Basaa Smith go to my downloads go to go to my.",
    "basaa Smith go to my downloads go to go to my.",
    "pictures of me when i was a baby",
    "i'm looking for the files i",
    "Locate savedpresentation2",
    "Locate savedpresentation3",
    '"Cortana, my friend sent me a saved document, where is it?"',
    "two thousand fourteen",
    # key word to remove
    # need to check in the future
    "computer",
    "phone",
    "folder",
    "desktop",
    "pc",
    "recycle bin",
    "notebook",
    "group post",
    "favorites",
    "winthrop",
    "disk",
    "device",
    "videos",
    "video",
    "cloud",
    "to do",
    "program files (x86)",
    "program file(x86)",
    "my wife",
    "my house",
    # for debug purpose
    # use regular expression to do in the futre
    "christmas list",
    "grocery list",
    "grocery store list",
    "ideas list",
    "music list",
    "to do list",
    "to-do list",
    "watch list",
    "walmart list",
    "wunderlist",
    "my list",
    "christmas playlist",
    "grocery playlist",
    "grocery store playlist",
    "ideas playlist",
    "music playlist",
    "to do playlist",
    "to-do playlist",
    "watch playlist",
    "walmart playlist",
    "my playlist",
    "christmas library",
    "grocery library",
    "grocery store library",
    "ideas library",
    "music library",
    "to do library",
    "to-do library",
    "watch library",
    "my library",
    "(name of spreadsheet)",
    "my lists",
    # not sure how to annotate, skip at first
    "Find my notes on Van's",
    "show my notes on group post",
    "downloads on this pc",
    "my pics",
    "go to my saved items",
    "what's on my list?",
    "what pictures on my phone",
    "show me the pictures i downloaded from my samsung",
    "open photos",
    "photos i have from winthrop",
    "open my photos",
    "find the note for wendy's",
    "cortana what music do i have",
    "what are my what's on my list today",
    "find me note about mario's",
    "find my spreadsheet in my documents",
    "go to last note last",
    "preview note with name chili's",
    "the pictures of december",
    "hey cortana show me photos that taken two thousand and sixteen",
    "find apk files that i",
    "Can you open my I was using files for me",
    "can you open my I was using files for me",
    "show my pictures of sunset from kim",
    "locate the letters i wrote to cecilia between march 27th , 2013 and september 9th , 2013",
    "thomson holidays my bookings my documents",
    }





fileTypeCandidateSet = set()
fileNameCandidateSet = set()
fileKeywordCandidateSet = set()
fileContactNameCandidateSet = set()
fileToContactNameCandidateSet = set()
orderRefCandidateSet = set()
#fileStartTimeCandidate = set()
fileTimeCandidateSet = set()
fileDataSourceCandidateSet = set()

# deduplication
skipQueryCandidateSet = set()


Output = [];

with codecs.open('files_mystuff.tsv', 'r', 'utf-8') as fin:
    for line in fin:
        line = line.strip();
        if not line:
            continue;
        linestrs = line.split("\t");
        # make sure it at least has
        # Query	ExternalFeature	Weight	Intent	Domain	Slot
        if len(linestrs) < 6:
            continue;

        skip = False
        # cannot use str
        # it will override existing function
        for blackListQuery in sorted (blackListQuerySet) :
            if linestrs[0].lower().find(blackListQuery.lower()) != -1:
                #print(linestrs[0])
                #skipQueryCandidate.add(linestrs[0])
                skip = True
                break

        if skip is True:
            skipQueryCandidateSet.add(linestrs[0])
            continue

        # make sure it is find_my_stuff intent
        if linestrs[3] in myStuffIntentToFileIntent:

            slot = linestrs[5]
            for key in sorted (myStuffSlotToFileSlot.keys()) :
                slot = slot.replace(key, myStuffSlotToFileSlot[key])

            # planning to have from_contact_name and contact_name at the same time
            # in this case, we will tage my my i I so no need this replacement
            #for key in sorted (removeSpecialSlotValue.keys()) :
            #    slot = slot.replace(key, removeSpecialSlotValue[key])
                
            # remove head and end spaces 
            slot = slot.strip()

            # for downloads for teams
            # pure to hae slot
            # ? might not work since it will double replacement when <> downloads </> exist
            #if slot.find("downloads") != -1:
            #    slot = slot.replace("downloads", "<file_type> downloads </file_type>")


            # fine-grained parse
            #list = re.findall("(</?[^>]*>)", slot)


            # handle my
            # to my
            if slot.find(" to my ") != -1:
                slot = slot.replace(" to my", " to <contact_name> my </contact_name>")
            # with me / to me 
            if slot.find(" to me ") != -1:
                slot = slot.replace(" to me", " to <to_contact_name> me </to_contact_name>")
            if slot.find(" with me ") != -1:
                slot = slot.replace(" with me ", " with <to_contact_name> me </to_contact_name>")

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
                     ])
            contactNames = set(["i",
                            "I",
                            ])
            for verb in verbsAlongWithContactName:
                for contactName in contactNames:
                    # with file action already
                    # try to tag contact name
                    # will suffer big / small case problems but leave it there eg: query small but annotation cbig
                    if linestrs[0].find(contactName +" "+ verb) != -1 and slot.find(contactName +" <file_action> "+verb+" </file_action>")!=-1:
                        slot = slot.replace(contactName +" <file_action> "+verb+" </file_action>", "<contact_name> "+contactName +" </contact_name>"+ " <file_action> "+verb+" </file_action>")
                    if linestrs[0].find(contactName +" was "+ verb) != -1 and slot.find(contactName +" was <file_action> "+verb+" </file_action>")!=-1:
                        slot = slot.replace(contactName +" was <file_action> "+verb+" </file_action>", "<contact_name> "+contactName +" </contact_name>"+ " was <file_action> "+verb+" </file_action>")

                    # with contact name already
                    # try to tag file action
                    # will suffer big / small case problems but leave it there eg: query small but annotation cbig
                    if linestrs[0].find(contactName +" "+ verb) != -1 and slot.find("<contact_name> "+ contactName + " </contact_name> "+verb)!=-1:
                        slot = slot.replace("<contact_name> "+ contactName + " </contact_name> "+verb, "<contact_name> "+contactName +" </contact_name>"+ " <file_action> "+verb+" </file_action>")
                    if linestrs[0].find(contactName +" was "+ verb) != -1 and slot.find("<contact_name> "+ contactName + " </contact_name> was "+verb)!=-1:
                        slot = slot.replace(contactName +" was <file_action> "+verb+" </file_action>", "<contact_name> "+contactName +" </contact_name>"+ " was <file_action> "+verb+" </file_action>")

            
                #if linestrs[0].find("i downloaded") != -1 and slot.find("i <file_action> downloaded </file_action>")!=-1:
                #    slot = slot.replace("i <file_action> downloaded </file_action>", "<contact_name> i </contact_name> <file_action> downloaded </file_action>")
            
                #if linestrs[0].find("I downloaded") != -1 and slot.find("I <file_action> downloaded </file_action>")!=-1:
                #    slot = slot.replace("i <file_action> downloaded </file_action>", "<contact_name> I </contact_name> <file_action> downloaded </file_action>")


            # hanlde my xxx and my should be contact name
            # ? not sure why this sniffet does not work
            ##nouns = [
            ##    "party"
            ##    ]
            ##for noun in nouns:
            ##    if slot.find("<keyword> my "+noun+" </keyword>")!= -1:
            ##        slot = slot.replace("<keyword> my " + noun +" </keyword>", "<contact_name> my </contact_name>"+" <keyword> "+ noun +" </keyword>")

            # tailor downloads special cases for teams
            # add new slot file_folder  for downloads
            if slot.find("<file_type> downloads </file_type>") != -1:
                slot = slot.replace("<file_type> downloads </file_type>", "<file_folder> downloads </file_folder>")
            if slot.find("<file_type> download </file_type>") != -1:
                slot = slot.replace("<file_type> download </file_type>", "<file_folder> download </file_folder>")

            # start time too long handle
            if slot.find("<time> in the last hour </time>") != -1:
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


            # after all slot replacement
            # remove head and end spaces 
            slot = slot.strip()

            # for analysis
            xmlpairs = re.findall("(<.*?>.*?<\/.*?>)", slot)
            #print (xmlpairs)
            for xmlpair in xmlpairs:
                
                if xmlpair.startswith("<file_type>"):
                    fileTypeCandidateSet.add(xmlpair)
                if xmlpair.startswith("<file_name>"):
                    fileNameCandidateSet.add(xmlpair)
                if xmlpair.startswith("<file_keyword>"):
                    fileKeywordCandidateSet.add(xmlpair)
                if xmlpair.startswith("<contact_name>"):
                    fileContactNameCandidateSet.add(xmlpair)
                if xmlpair.startswith("<to_contact_name>"):
                    fileToContactNameCandidateSet.add(xmlpair)
                if xmlpair.startswith("<order_ref>"):
                    orderRefCandidateSet.add(xmlpair)
                if xmlpair.startswith("<time>"):
                    fileTimeCandidateSet.add(xmlpair)
                if xmlpair.startswith("<data_source>"):
                    fileDataSourceCandidateSet.add(xmlpair)
            
            # output id	query	intent	domain	QueryXml	id	0   
            Output.append("0\t"+linestrs[0]+"\t"+myStuffIntentToFileIntent[linestrs[3]]+"\t"+myStuffDomainToFileDomain[linestrs[4]]+"\t"+slot);

"""
# comment shuffle in the first place
#random.shuffle(OutputSet);
"""

with codecs.open('files_mystuff_after_filtering.tsv', 'w', 'utf-8') as fout:
    for item in Output:
        fout.write(item + '\r\n');

with codecs.open('files_skip_query.tsv', 'w', 'utf-8') as fout:
    for item in skipQueryCandidateSet:
        fout.write(item + '\r\n');

#######################
# slot level output
#######################


with codecs.open('files_mystuff_after_filtering_file_type.tsv', 'w', 'utf-8') as fout:
    for item in fileTypeCandidateSet:
        fout.write(item + '\r\n');

    
with codecs.open('files_mystuff_after_filtering_file_keyword.tsv', 'w', 'utf-8') as fout:
    for item in fileKeywordCandidateSet:
        fout.write(item + '\r\n');

with codecs.open('files_mystuff_after_filtering_file_name.tsv', 'w', 'utf-8') as fout:
    for item in fileNameCandidateSet:
        fout.write(item + '\r\n');

with codecs.open('files_mystuff_after_filtering_contact_name.tsv', 'w', 'utf-8') as fout:
    for item in fileContactNameCandidateSet:
        fout.write(item + '\r\n');

with codecs.open('files_mystuff_after_filtering_to_contact_name.tsv', 'w', 'utf-8') as fout:
    for item in fileToContactNameCandidateSet:
        fout.write(item + '\r\n');

with codecs.open('files_mystuff_after_filtering_order_ref.tsv', 'w', 'utf-8') as fout:
    for item in orderRefCandidateSet:
        fout.write(item + '\r\n');

with codecs.open('files_mystuff_after_filtering_time.tsv', 'w', 'utf-8') as fout:
    for item in fileTimeCandidateSet:
        fout.write(item + '\r\n');

with codecs.open('files_mystuff_after_filtering_data_source.tsv', 'w', 'utf-8') as fout:
    for item in fileDataSourceCandidateSet:
        fout.write(item + '\r\n');

#######################
# query replacement revert
#######################
'''
RefineSet = [];
with codecs.open('files_mystuff.tsv', 'r', 'utf-8') as fin:
    for line in fin:
        line = line.strip();
        if not line:
            continue;
        linestrs = line.split("\t");
        # make sure it at least has
        # Query	ExternalFeature	Weight	Intent	Domain	Slot
        if len(linestrs) < 6:
            continue;
        # for query replacement bug
        # it will not update original datasets
        linestrs[0] = linestrs[0].replace("<file action> working </file_action>", "working")


        singleLine = ""
        for str in linestrs:
            if len(singleLine) == 0:
                singleLine = singleLine + str
            else:
                singleLine = singleLine + "\t"+str
            
        
        RefineSet.append(singleLine);

with codecs.open('files_mystuff_revert.tsv', 'w', 'utf-8') as fout:
    for item in RefineSet:
        fout.write(item + '\r\n');
'''

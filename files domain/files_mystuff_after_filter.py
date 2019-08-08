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
    #"<start_date>" : "<start_date>",
    #"</start_date>" : "</start_date>",
    #"<start_time>" : "<start_time>",
    #"</start_time>" : "</start_time>",
    #"<end_date>" : "<end_date>",
    #"</end_date>" : "</end_date>",
    #"<end_time>" : "<end_time>",
    #"</end_time>" : "</end_time>",
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
    }



fileTypeCandidate = []
fileNameCandidate = []
fileKeywordCandidate = []
fileContactNameCandidate = []
fileToContactNameCandidate = []
fileOrderRefCandidate = []
fileStartTimeCandidate = []
fileDataSourceCandidate = []


# deduplication
skipQueryCandidate = set()


OutputSet = [];

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
            skipQueryCandidate.add(linestrs[0])
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
            if slot.find("to my") != -1:
                slot = slot.replace("to my", "to <contact_name> my </contact_name>")
            # with me
            if slot.find("to me") != -1:
                slot = slot.replace("to me", "to <to_contact_name> me </to_contact_name>")
            if slot.find("with me") != -1:
                slot = slot.replace("with me", "with <to_contact_name> me </to_contact_name>")

            # i verb
            verbs = ["downloaded",
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
                     ]
            contactNames = ["i",
                            "I"
                            ]
            for verb in verbs:
                for contactName in contactNames:
                    # tag contact name
                    if linestrs[0].find(contactName +" "+ verb) != -1 and slot.find(contactName +" <file_action> "+verb+" </file_action>")!=-1:
                        slot = slot.replace(contactName +" <file_action> "+verb+" </file_action>", "<contact_name> "+contactName +" </contact_name>"+ " <file_action> "+verb+" </file_action>")
                    if linestrs[0].find(contactName +" was "+ verb) != -1 and slot.find(contactName +" was <file_action> "+verb+" </file_action>")!=-1:
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
            if slot.find("<data_source> downloads </data_source>") != -1:
                slot = slot.replace("<data_source> downloads </data_source>", "<file_type> downloads </file_type>")

            # start time too long handle
            if slot.find("<start_time> in the last hour </start_time>") != -1:
                slot = slot.replace("<start_time> in the last hour </start_time>", "in the <start_time> last hour </start_time>")
            if slot.find("<start_time> within the last hour </start_time>") != -1:
                slot = slot.replace("<start_time> within the last hour </start_time>", "within the <start_time> last hour </start_time>")

            if slot.find("<start_time> in the last day </start_time>") != -1:
                slot = slot.replace("<start_time> in the last day </start_time>", "in the <start_time> last day </start_time>")
            if slot.find("<start_time> within the last day </start_time>") != -1:
                slot = slot.replace("<start_time> within the last day </start_time>", "within the <start_time> last day </start_time>")


            if slot.find("<start_date> in the last month </start_date>") != -1:
                slot = slot.replace("<start_date> in the last month </start_date>", "in the <start_date> last month </start_date>")
            if slot.find("<start_date> within the last 1 month </start_date>") != -1:
                slot = slot.replace("<start_date> within the last month </start_date>", "within the <start_date> last month </start_date>")


            if slot.find("<start_date> in the last week </start_date>") != -1:
                slot = slot.replace("<start_date> in the last week </start_date>", "in the <start_date> last week </start_date>")
            if slot.find("<start_date> within the last week </start_date>") != -1:
                slot = slot.replace("<start_date> within the last 1 week </start_date>", "within the <start_date> last week </start_date>")


            #https://blog.csdn.net/blueheart20/article/details/52883045
            for num in range(1, 400):
                if slot.find("<start_date> in the last " + str(num) + " hours </start_date>") != -1:
                    slot = slot.replace("<start_date> in the last " + str(num) + " hours </start_date>", "in the <start_date> last " + str(num) + " hours </start_date>")
                if slot.find("<start_date> within the last " + str(num) + " hours </start_date>") != -1:
                    slot = slot.replace("<start_date> within the last " + str(num) + " hours </start_date>", "within the <start_date> last " + str(num) + " hours </start_date>")

                if slot.find("<start_date> in the last " + str(num) + " days </start_date>") != -1:
                    slot = slot.replace("<start_date> in the last " + str(num) + " days </start_date>", "in the <start_date> last " + str(num) + " days </start_date>")
                if slot.find("<start_date> within the last " + str(num) + " days </start_date>") != -1:
                    slot = slot.replace("<start_date> within the last " + str(num) + " days </start_date>", "within the <start_date> last " + str(num) + " days </start_date>")

                
                if slot.find("<start_date> in the last " + str(num) + " months </start_date>") != -1:
                    slot = slot.replace("<start_date> in the last " + str(num) + " months </start_date>", "in the <start_date> last " + str(num) + " months </start_date>")
                if slot.find("<start_date> within the last " + str(num) + " months </start_date>") != -1:
                    slot = slot.replace("<start_date> within the last " + str(num) + " months </start_date>", "within the <start_date> last " + str(num) + " months </start_date>")

                if slot.find("<start_date> in the last " + str(num) + " weeks </start_date>") != -1:
                    slot = slot.replace("<start_date> in the last " + str(num) + " weeks </start_date>", "in the <start_date> last " + str(num) + " weeks </start_date>")
                if slot.find("<start_date> within the last " + str(num) + " weeks </start_date>") != -1:
                    slot = slot.replace("<start_date> within the last " + str(num) + " weeks </start_date>", "within the <start_date> last " + str(num) + " weeks </start_date>")
                    
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
                if slot.find("<start_date> in the last " + alphadigit + " hours </start_date>") != -1:
                    slot = slot.replace("<start_date> in the last " + alphadigit + " hours </start_date>", "in the <start_date> last " + alphadigit + " hours </start_date>")
                if slot.find("<start_date> within the last " + alphadigit + " hours </start_date>") != -1:
                    slot = slot.replace("<start_date> within the last " + alphadigit + " hours </start_date>", "within the <start_date> last " + alphadigit + " hours </start_date>")

                if slot.find("<start_date> in the last " + alphadigit + " days </start_date>") != -1:
                    slot = slot.replace("<start_date> in the last " + alphadigit + " days </start_date>", "in the <start_date> last " + alphadigit + " days </start_date>")
                if slot.find("<start_date> within the last " + alphadigit + " days </start_date>") != -1:
                    slot = slot.replace("<start_date> within the last " + alphadigit + " days </start_date>", "within the <start_date> last " + alphadigit + " days </start_date>")
                    
                if slot.find("<start_date> in the last " + alphadigit + " months </start_date>") != -1:
                    slot = slot.replace("<start_date> in the last " + alphadigit + " months </start_date>", "in the <start_date> last " + alphadigit + " months </start_date>")
                if slot.find("<start_date> within the last " + alphadigit + " months </start_date>") != -1:
                    slot = slot.replace("<start_date> within the last " + alphadigit + " months </start_date>", "within the <start_date> last " + alphadigit + " months </start_date>")

                if slot.find("<start_date> in the last " + alphadigit + " weeks </start_date>") != -1:
                    slot = slot.replace("<start_date> in the last " + alphadigit + " weeks </start_date>", "in the <start_date> last " + alphadigit + " weeks </start_date>")
                if slot.find("<start_date> within the last " + alphadigit + " weeks </start_date>") != -1:
                    slot = slot.replace("<start_date> within the last " + alphadigit + " weeks </start_date>", "within the <start_date> last " + alphadigit + " weeks </start_date>")


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
            if slot.find("<start_time> recent </start_time>") != -1:
                slot = slot.replace("<start_time> recent </start_time>", "<file_recency> recent </file_recency>")
            if slot.find("<start_time> Recent </start_time>") != -1:
                slot = slot.replace("<start_time> Recent </start_time>", "<file_recency> Recent </file_recency>")
            if slot.find("<start_time> recently </start_time>") != -1:
                slot = slot.replace("<start_time> recently </start_time>", "<file_recency> recently </file_recency>")
            if slot.find("<start_time> Recently </start_time>") != -1:
                slot = slot.replace("<start_time> Recently </start_time>", "<file_recency> Recently </file_recency>")
            if slot.find("<start_time> just </start_time>") != -1:
                slot = slot.replace("<start_time> just </start_time>", "<file_recency> just </file_recency>")
            if slot.find("<start_time> Just </start_time>") != -1:
                slot = slot.replace("<start_time> Just </start_time>", "<file_recency> Just </file_recency>")
            if slot.find("<order_ref> most recent </order_ref>") != -1:
                slot = slot.replace("<order_ref> most recent </order_ref>", "most <file_recency> recent </file_recency>")

            # for analysis
            xmlpairs = re.findall("(<.*?>.*?<\/.*?>)", slot)
            #print (xmlpairs)
            for xmlpair in xmlpairs:
                
                if xmlpair.startswith("<file_type>"):
                    fileTypeCandidate.append(xmlpair)
                if xmlpair.startswith("<file_name>"):
                    fileNameCandidate.append(xmlpair)
                if xmlpair.startswith("<file_keyword>"):
                    fileKeywordCandidate.append(xmlpair)
                if xmlpair.startswith("<contact_name>"):
                    fileContactNameCandidate.append(xmlpair)
                if xmlpair.startswith("<to_contact_name>"):
                    fileToContactNameCandidate.append(xmlpair)
                if xmlpair.startswith("<order_ref>"):
                    fileOrderRefCandidate.append(xmlpair)
                if xmlpair.startswith("<start_time>"):
                    fileStartTimeCandidate.append(xmlpair)
                if xmlpair.startswith("<data_source>"):
                    fileDataSourceCandidate.append(xmlpair)
            
            # output id	query	intent	domain	QueryXml	id	0   
            OutputSet.append("0\t"+linestrs[0]+"\t"+myStuffIntentToFileIntent[linestrs[3]]+"\t"+myStuffDomainToFileDomain[linestrs[4]]+"\t"+slot);

"""
# comment shuffle in the first place
#random.shuffle(OutputSet);
"""

with codecs.open('files_mystuff_after_filtering.tsv', 'w', 'utf-8') as fout:
    for item in OutputSet:
        fout.write(item + '\r\n');


with codecs.open('files_mystuff_after_filtering_file_type.tsv', 'w', 'utf-8') as fout:
    for item in fileTypeCandidate:
        fout.write(item + '\r\n');

    
with codecs.open('files_mystuff_after_filtering_file_keyword.tsv', 'w', 'utf-8') as fout:
    for item in fileKeywordCandidate:
        fout.write(item + '\r\n');

with codecs.open('files_mystuff_after_filtering_file_name.tsv', 'w', 'utf-8') as fout:
    for item in fileNameCandidate:
        fout.write(item + '\r\n');

with codecs.open('files_mystuff_after_filtering_contact_name.tsv', 'w', 'utf-8') as fout:
    for item in fileContactNameCandidate:
        fout.write(item + '\r\n');

with codecs.open('files_mystuff_after_filtering_to_contact_name.tsv', 'w', 'utf-8') as fout:
    for item in fileToContactNameCandidate:
        fout.write(item + '\r\n');

with codecs.open('files_mystuff_after_filtering_order_ref.tsv', 'w', 'utf-8') as fout:
    for item in fileOrderRefCandidate:
        fout.write(item + '\r\n');

with codecs.open('files_mystuff_after_filtering_start_time.tsv', 'w', 'utf-8') as fout:
    for item in fileStartTimeCandidate:
        fout.write(item + '\r\n');

with codecs.open('files_mystuff_after_filtering_data_source.tsv', 'w', 'utf-8') as fout:
    for item in fileDataSourceCandidate:
        fout.write(item + '\r\n');


with codecs.open('files_skip_query.tsv', 'w', 'utf-8') as fout:
    for item in skipQueryCandidate:
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

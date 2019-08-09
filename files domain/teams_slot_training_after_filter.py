import codecs;
import random;
import re;


import math
import re
import sys

# add hyper paramter if unbalanced
hyper_parameter = 200



fileDomainRelatedIntent = ['file_search', 'file_open', 'file_share', 'file_download', 'file_other', 'file_navigate']

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



blackListQuerySet = {
    }



fileKeywordCandidate = []
meetingStarttimeCandidate = []
fileTypeCandidate = []
fileRecencyCandidate = []
sharetargetTypeCandidate = []
sharetargetNameCandidate = []
contactNameCandidate =[]
 

# deduplication
skipQueryCandidate = set()


OutputSet = [];

with codecs.open('teams_slot_training.tsv', 'r', 'utf-8') as fin:
    for line in fin:
        line = line.strip();
        if not line:
            continue;
        linestrs = line.split("\t");
        # make sure it at least has
        # Query	ExternalFeature	Weight	Intent	Domain	Slot
        if len(linestrs) < 5:
            continue;

        # make sure it is find_my_stuff intent
        if linestrs[2] in fileDomainRelatedIntent:

            slot = linestrs[4]
            for key in sorted (teamsSlotToFileSlot.keys()) :
                slot = slot.replace(key, teamsSlotToFileSlot[key])

            # planning to have from_contact_name and contact_name at the same time
            # in this case, we will tage my my i I so no need this replacement
            #for key in sorted (removeSpecialSlotValue.keys()) :
            #    slot = slot.replace(key, removeSpecialSlotValue[key])
                
            # remove head and end spaces 
            slot = slot.strip()

            # fine-grained parse
            #list = re.findall("(</?[^>]*>)", slot)


            # handle my
            # to my
            '''
            if slot.find(" to my ") != -1:
                slot = slot.replace(" to my", " to <contact_name> my </contact_name>")
            # with me
            if slot.find(" to me ") != -1:
                slot = slot.replace(" to me", " to <to_contact_name> me </to_contact_name>")
            if slot.find(" with me ") != -1:
                slot = slot.replace(" with me ", " with <to_contact_name> me </to_contact_name>")

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
            '''

            # for analysis
            xmlpairs = re.findall("(<.*?>.*?<\/.*?>)", slot)
            #print (xmlpairs)
            for xmlpair in xmlpairs:
                
                if xmlpair.startswith("<file_keyword>"):
                    fileKeywordCandidate.append(xmlpair)
                if xmlpair.startswith("<meeting_starttime>"):
                    meetingStarttimeCandidate.append(xmlpair)
                if xmlpair.startswith("<file_type>"):
                    fileTypeCandidate.append(xmlpair)
                if xmlpair.startswith("<file_recency>"):
                    fileRecencyCandidate.append(xmlpair)
                if xmlpair.startswith("<sharetarget_type>"):
                    sharetargetTypeCandidate.append(xmlpair)
                if xmlpair.startswith("<sharetarget_name>"):
                    sharetargetNameCandidate.append(xmlpair)
                if xmlpair.startswith("<contact_name>"):
                    contactNameCandidate.append(xmlpair)

            # output: id	query	intent	domain	QueryXml	id	0   
            OutputSet.append(linestrs[0]+"\t"+linestrs[1]+"\t"+linestrs[2]+"\t"+teamsDomainToFileDomain[linestrs[3]]+"\t"+slot);

"""
# comment shuffle in the first place
#random.shuffle(OutputSet);
"""

with codecs.open('teams_slot_training_after_filtering.tsv', 'w', 'utf-8') as fout:
    for item in OutputSet:
        fout.write(item + '\r\n');

with codecs.open('teams_slot_training_after_filtering_file_keyword.tsv', 'w', 'utf-8') as fout:
    for item in fileKeywordCandidate:
        fout.write(item + '\r\n');

with codecs.open('teams_slot_training_after_filtering_meeting_starttime.tsv', 'w', 'utf-8') as fout:
    for item in meetingStarttimeCandidate:
        fout.write(item + '\r\n');

with codecs.open('teams_slot_training_after_filtering_file_type.tsv', 'w', 'utf-8') as fout:
    for item in fileTypeCandidate:
        fout.write(item + '\r\n');

with codecs.open('teams_slot_training_after_filtering_file_recency.tsv', 'w', 'utf-8') as fout:
    for item in fileRecencyCandidate:
        fout.write(item + '\r\n');

with codecs.open('teams_slot_training_after_filtering_sharetarget_type.tsv', 'w', 'utf-8') as fout:
    for item in sharetargetTypeCandidate:
        fout.write(item + '\r\n');

with codecs.open('teams_slot_training_after_filtering_sharetarget_name.tsv', 'w', 'utf-8') as fout:
    for item in sharetargetNameCandidate:
        fout.write(item + '\r\n');

with codecs.open('teams_slot_training_after_filtering_contact_name.tsv', 'w', 'utf-8') as fout:
    for item in contactNameCandidate:
        fout.write(item + '\r\n');


#######################
# query replacement revert
#######################

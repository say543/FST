import codecs;
import random;
import re;


import math
import re
import sys

# add hyper paramter if unbalanced
hyper_parameter = 200




PREVIOUSTURNDOMAIN = "PreviousTurnDomain"
PREVIOUSTURNINTENT = "PreviousTurnIntent"
TASKFRAMESTATUS = "TaskFrameStatus"
TASKFRAMEENTITYSTATES = "TaskFrameEntityStates"
TASKFRAMEGUID = "TaskFrameGUID"
SPEECHPEOPLEDISAMBIGUATIONGRAMMARMATCHES = "SpeechPeopleDisambiguationGrammarMatches"
CONVERSATIONALCONTEXT = "ConversationalContext"


commonDomainRelatedIntent = ['cancel','confirm','reject','select_none']

#inmeetingDomainToFileDomain = {
#    "inmeeting" : "files"
#}

wrongcontactname = set([
    'frisbee',
    'salsa',
    'deck',
    'now',
    'sure',
    'kinder'
    ])


lineNumber = 0;
with codecs.open('lexicon.calendar.person_names_for_training.txt', 'r', 'utf-8') as fin:

    print('####start####')
    print(fin.name)
    print('####start####')
    for line in fin:
        line = line.strip();
        if not line:
            continue;

        if line.lower() in wrongcontactname:
            print(line)
            print(lineNumber)

        lineNumber+=1

lineNumber = 0;
# will encounter utf-8 issues
# so igmore error
#with codecs.open('calendar_peopleNames.20180119_used_for_some_patten.txt', 'r', 'utf-8') as fin:
with codecs.open('calendar_peopleNames.20180119_used_for_some_patten.txt', 'r', 'utf-8',errors='ignore') as fin:


    print('####start####')
    print(fin.name)
    print('####start####')
    for line in fin:
        line = line.strip();
        if not line:
            continue;

        #for debug
        #print(line)

        if line.lower() in wrongcontactname:
            print(line)
            print(lineNumber)

        lineNumber+=1



'''
comment at first not a one being used for training or pattern
lineNumber = 0;
with codecs.open('calendar_contactName_FST_pattern.txt', 'r', 'utf-8') as fin:

    print('####start####')
    print(fin.name)
    print('####start####')
    for line in fin:
        line = line.strip();
        if not line:
            continue;

        if line.lower() in wrongcontactname:
            print(line)
            print(lineNumber)

        lineNumber+=1
'''     
            

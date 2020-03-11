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
    'deck'
    ])


lineNumber = 0;
with codecs.open('lexicon.calendar.person_names_for_training.txt', 'r', 'utf-8') as fin:
    for line in fin:
        line = line.strip();
        if not line:
            continue;

        if line in wrongcontactname:
            print(line)
            print(lineNumber)

        lineNumber+=1
            

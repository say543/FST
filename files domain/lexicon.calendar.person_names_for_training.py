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
    'kinder',
    'pamplin',
    'greenland',
    'skills',
    'spring'
    ])


###########
### people lexicon preprocessing routine, uncomment / comment based on your change
###########
peopleOriginalSet = set([]);
peopleOutputsSetAfterPreprocessing = set([]);
peopleLexiconFile = 'people_NameListMSFT.txt'
with codecs.open('people_NameListMSFT.txt', 'r', 'utf-8') as fin:

    print('####preprocessing ####')
    print(fin.name)
    print('####preprocessing ####')
    
    for line in fin:
        line = line.strip();
        if not line:
            continue;

        peopleOriginalSet.add(line.lower())

        array = line.split(' ');
        for i in range(len(array)):
            peopleOutputsSetAfterPreprocessing.add(array[i].lower())


### if want to output peopleOutputsSetAfterPreprocessing
# turning on
#print(len(outputsSet))
#with codecs.open((peopleLexiconFile.split("."))[0] +'_preprocessing.tsv', 'w', 'utf-8') as fout:
#    for item in sorted(outputsSet):
#        fout.write(item + '\r\n');


            
trainingAndPatternCombineSet= set([]);


trainingLexiconFile = 'lexicon.calendar.person_names_for_training.txt'
lineNumber = 0;
with codecs.open(trainingLexiconFile, 'r', 'utf-8') as fin:

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
        else:
            trainingAndPatternCombineSet.add(line.lower())

        lineNumber+=1


patternLexiconFile = 'calendar_peopleNames.20180119_used_for_some_patten.txt'
lineNumber = 0;
# will encounter utf-8 issues
# so igmore error
#with codecs.open('calendar_peopleNames.20180119_used_for_some_patten.txt', 'r', 'utf-8') as fin:
with codecs.open(patternLexiconFile, 'r', 'utf-8',errors='ignore') as fin:


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
        else:
            trainingAndPatternCombineSet.add(line.lower())

        lineNumber+=1






# step1:
# filter trainingAndPatternCombineSet by people
#peopleOriginalSet = set([]);
#peopleOutputsSetAfterPreprocessing = set([]);
# all input are lower cases already
trainingAndPatternCombineSetAfterFilter= set([]);
trainingAndPatternCombineSetBeingFilter = set([]);


for lexicon in trainingAndPatternCombineSet:
    if lexicon in peopleOriginalSet or lexicon in peopleOutputsSetAfterPreprocessing:
        trainingAndPatternCombineSetAfterFilter.add(lexicon)
    else:
        trainingAndPatternCombineSetBeingFilter.add(lexicon)

#step2
#augmentation based on single word
#peopleOutputsSetAfterPreprocessing = set([]);
for lexicon in peopleOriginalSet:
    trainingAndPatternCombineSetAfterFilter.add(lexicon)



trainingAndPatternCombineSetAfterFilterAndAugmentFile = 'combine_lexicon.txt'

with codecs.open(trainingAndPatternCombineSetAfterFilterAndAugmentFile, 'w', 'utf-8') as fout:

    print('####start####')
    print(trainingAndPatternCombineSetAfterFilterAndAugmentFile)
    print(len(trainingAndPatternCombineSetAfterFilter))
    print('####start####')
    for item in sorted(trainingAndPatternCombineSetAfterFilter):
        fout.write(item + '\r\n');


# for debug
with codecs.open((trainingAndPatternCombineSetAfterFilterAndAugmentFile.split("."))[0] +'_deprecated.tsv', 'w', 'utf-8') as fout:

    print('####start####')
    print((trainingAndPatternCombineSetAfterFilterAndAugmentFile.split("."))[0] +'_deprecated.tsv')
    print(len(trainingAndPatternCombineSetBeingFilter))
    print('####start####')
    
    for item in sorted(trainingAndPatternCombineSetBeingFilter):
        fout.write(item + '\r\n');



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
            

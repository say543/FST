import codecs;
import random;
import re;
from collections import defaultdict;
import os;
from shutil import copyfile


import math
import re
import sys

# 1: slot only
# 2: intent only
# 5: both
synthetic_mode = 1;


# in kashan
# 0.3 from zhiguo real suer file name
# 0.1 from teams originla file_keyword
# 0.6 generated file name (my search term can be this)

# add hyper paramter if unbalanced
# following people
# one pattern generate one but duplicate time
#p from [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] 
#N from [ 20, 50, 100, 200, 300, 400, 500, 600, 700]
# starting with 300
# limit each one to repeast 20 only since more pattern
sampling_hyper_paramter_each_slot = 20
#sampling_hyper_paramter_each_slot = 6000
#sampling_hyper_paramter_each_slot = 9000
# for intent using to prevent too many



#numberofQuery_hyper_parameter = 200
#numberofQuery_hyper_parameter = 400
numberofQuery_hyper_parameter = 100
#numberofQuery_hyper_parameter = 50
#numberofQuery_hyper_parameter = 25



#numberofQuery_hyper_parameter = 6000
#numberofQuery_hyper_parameter = 9000

# for slot using 20
#numberofQuery_hyper_parameter = 40

# multiple holder randomization
multiple_holder_hyperparameter = 3



# random seed	
# and inner loop random seed change	
rand_seed_parameter_initialization = 0.1	
rand_seed_offset = 0.01

PREVIOUSTURNDOMAIN = "PreviousTurnDomain"	
PREVIOUSTURNINTENT = "PreviousTurnIntent"	
TASKFRAMESTATUS = "TaskFrameStatus"	
TASKFRAMEENTITYSTATES = "TaskFrameEntityStates"	
TASKFRAMEGUID = "TaskFrameGUID"	
SPEECHPEOPLEDISAMBIGUATIONGRAMMARMATCHES = "SpeechPeopleDisambiguationGrammarMatches"	
CONVERSATIONALCONTEXT = "ConversationalContext"


def getListForType(filename):
    # read open text content
    holderList = [];
    with codecs.open(filename, 'r', 'utf-8') as fin:
        for line in fin:
            line = line.strip();
            if not line:
                continue;
            holderList.append(line);
    return holderList

def samleOpenText(holderList, HolderName, slotListDictionary):
    # sample the open text
    wordList = random.sample(holderList, k=min(sampling_hyper_paramter_each_slot, len(holderList)));
    #print(wordList)
    slotListDictionary[HolderName] = wordList;

def parse(slotList, doubleSlotList):
    # slot / holder order needs to be consistent
    numSlots = len(slotList);
    HolderName = [];

    # for each slot, replace specila chracter and use uper for place holder
    for slot in slotList:
        HolderName.append(slot.replace('_', '').upper() + 'HOLDER');

    MultipleHolderName = set()
    # for each slot, replace specila chracter and use uper for place holder
    for slot in doubleSlotList:
        MultipleHolderName.add(slot.replace('_', '').upper() + 'HOLDER');

    # read pattern definitions
    patternSet = [];
    with codecs.open('pattern.txt', 'r', 'utf-8') as fin:
        for line in fin:
            line = line.strip();
            if not line:
                continue;

            if line.startswith("#"):
                continue;

            # for deubg
            #print(line)
            
            patternSet.append(line);


    # prepare for each type of place holder
    #fileTypeList = getListForType("..\\file_type.txt")
    #contactNameTypeList = getListForType("..\\calendar_peopleNames.20180119_used_for_some_patten.txt")

    #slotListDictionary = defaultdict(list);
    #samleOpenText(fileTypeList, "file_type".replace('_', '').upper() + 'HOLDER',  slotListDictionary)
    #samleOpenText(contactNameTypeList, "contact_name".replace('_', '').upper() + 'HOLDER',  slotListDictionary)

    slotListDictionary = defaultdict(list);
    for slot, slotHolderName in zip(slotList, HolderName):
        # for debug
        print (slot)
        #typeList = getListForType("..\\"+slot+".txt")
        typeList = getListForType(slot+".txt")
        samleOpenText(typeList, slot.replace('_', '').upper() + 'HOLDER', slotListDictionary)

    #initial rand seed	
    rand_seed_parameter = rand_seed_parameter_initialization

    
    outputSet = [];
    outputIntentSet = [];
    for pattern in patternSet:

	# update seed fpr each pattern
	# initla version is determined so comment
        ##rand_seed_parameter=rand_seed_parameter+rand_seed_offset;	
        ##random.seed(rand_seed_parameter);

        #for debug
        #print(pattern)

        
        array = pattern.split('\t');

        # verfiy how many columns they are
        if len(array) < 5:
            continue;
        
        originalQuery = array[1];
        intent = array[2];
        domain = array[3];
        originalSlotXml = array[4];

	# prevent duplication so use low bound of numberofQuery_hyper_parameter / sampling_hyper_paramter_each_slot	
        for idx in range(numberofQuery_hyper_parameter):
            sampling_hyper_paramter_each_slot
            query = originalQuery;
            slotXml = originalSlotXml;

            # for debug
            #print(query)

	    # update seed fpr each query
	    # initla version is determined so comment
            #rand_seed_parameter=rand_seed_parameter+rand_seed_offset;	
            #random.seed(rand_seed_parameter);

            # key XXXHOLDER
            # value : something which should be inside <XMLTAG> </XMLTAG>
            for key, value in slotListDictionary.items():
                if key in query:
                    if key in MultipleHolderName:

                        # no multiple holder so this does not matter
                        cnt = random.randint(0, multiple_holder_hyperparameter)+1

                        combineValue = ""
                        for i in range(0,cnt):
                            inRangeIndex = random.randint(0, len(value)-1)
                            combineValue += " "+ value[inRangeIndex];

                        #print(combineValue)
                        combineValue = combineValue.strip()
                        query = query.replace(key, combineValue);
                        slotXml = slotXml.replace(key, combineValue);                        
                        
                    else:

                        # here still introduce filekeywrod randomization
                        inRangeIndex = random.randint(0, len(value)-1)

                        query = query.replace(key, value[inRangeIndex]);
                        
                        slotXml = slotXml.replace(key, value[inRangeIndex]);


                        
                        #for debug



                        #print(query)
            for n in range(sampling_hyper_paramter_each_slot):
                outputSet.append('\t'.join(['0', query, intent, domain, slotXml]));
                # add	
                # "PreviousTurnDomain"	
                # "PreviousTurnIntent"	
                # as	
                # 'TurnNumber', PREVIOUSTURNINTENT, 'query', 'intent',PREVIOUSTURNDOMAIN])	
                # append empty at first	
                #outputIntentSet.append('\t'.join(['0',"\t"+query,intent]));	
                outputIntentSet.append('\t'.join(['0',"\t"+query,intent,""]));

    outputSlot = '_'.join(slotList);

    
    if synthetic_mode == 1 or synthetic_mode == 5:
        # for slot
        print("generating domain synthetic...")
        #with codecs.open('domain_synthesised_' + 'n_'+ str(numberofQuery_hyper_parameter) + '.tsv', 'w', 'utf-8') as fout:
        with codecs.open('domain_synthesised' +'.tsv', 'w', 'utf-8') as fout:            
            for item in outputSet:
                fout.write(item + '\r\n');

        # for record, using .txt as extention so it will not being picked up
        with codecs.open('domain_synthesised_' + 'n_'+ str(numberofQuery_hyper_parameter) + '_r_' + str(sampling_hyper_paramter_each_slot) +'.txt', 'w', 'utf-8') as fout:         
            for item in outputSet:
                fout.write(item + '\r\n');        
    

    '''
    if synthetic_mode == 2 or synthetic_mode == 5:
        # for intent
        print("generating domain synthetic...")
        with codecs.open('domain_data_synthesised_' + outputSlot + '.tsv', 'w', 'utf-8') as fout:
            fout.write('\t'.join(['TurnNumber', PREVIOUSTURNINTENT, 'query', 'intent',PREVIOUSTURNDOMAIN, TASKFRAMESTATUS, TASKFRAMEENTITYSTATES, TASKFRAMEGUID, SPEECHPEOPLEDISAMBI4GUATIONGRAMMARMATCHES, CONVERSATIONALCONTEXT])+'\r\n');	
            for item in outputIntentSet:
                # for debug
                #print(item)
                
                fout.write(item + '\r\n');    
    '''

if __name__ == '__main__':

    # option 1
    # use search term to replace file_keyword and dedup
    '''
    copyfile("..\\..\LexiconFiles\\kaggle_searchterm_lexicon.txt" , "file_keyword.txt")
    copyfile("..\\..\LexiconFiles\\kaggle_searchterm_lexicon.txt" , "file_name.txt")
    copyfile("..\\..\LexiconFiles\\kaggle_searchterm_lexicon.txt" , "sharetarget_name.txt")
    '''


    # optn 2
    # use search term to replace file_keyword and dedup
    # also filter by <= 3 gram
    '''
    copyfile("..\\..\LexiconFiles\\kaggle_searchterm_lexicon.txt", "temp.txt");
    dedup = set()
    with codecs.open('temp.txt', 'r', 'utf-8') as fin:

        for line in fin:
            line = line.strip();
            if not line:
                continue;

            array = line.split()

            # only three gram features so limit to short
            if len(array) <=3: 
                dedup.add(line)

    with codecs.open('temp2.txt', 'w', 'utf-8') as fout:
        for item in sorted(dedup):
            fout.write(item + '\r\n');

    copyfile("temp2.txt" , "file_keyword.txt")
    copyfile("temp2.txt" , "file_name.txt")
    copyfile("temp2.txt" , "sharetarget_name.txt")
    '''


    # option 3
    # using original open text synthetic copy directly

    



    
    
    
    
    #slotList = ['contact_name', 'file_type', 'to_contact_name', "file_keyword"];
    #doubleSlotList = ['contact_name', 'to_contact_name']

    #tocontactname needs to be done before contactname since contactname is substring
    # 
    #slotList = ["file_keyword", 'to_contact_name', 'file_type', 'contact_name'];


    # in the future
    # data source (or file_boost )  needs to have attachment
    
    
    slotList = ["file_keyword", 'file_name', 'date', 'to_contact_name', 'file_type',  'file_boost', 'contact_name', "file_action", 'order_ref', 'file_recency', 'sharetarget_type', 'sharetarget_name','meeting_starttime', 'data_source'];

    #no need to have double slot any more since original has double lexicon
    #doubleSlotList = ['to_contact_name', 'contact_name']
    doubleSlotList = ['', '']

    
    #parse(slotList, hyper_parameter);
    parse(slotList, doubleSlotList);

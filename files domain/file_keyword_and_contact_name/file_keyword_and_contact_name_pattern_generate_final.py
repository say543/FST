import codecs;
import random;
from collections import defaultdict;

# 1: slot only
# 2: intent only
# 5: both
synthetic_mode = 1;

# add hyper paramter if unbalanced
hyper_parameter = 200


# multiple holder randomization
multiple_holder_hyperparameter = 3

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
    wordList = random.sample(holderList, k=min(hyper_parameter, len(holderList)));
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

            # skip comment with #
            if line.startswith('#'):
                continue
            
            if not line:
                continue;
            patternSet.append(line);


    # prepare for each type of place holder
    #fileTypeList = getListForType("..\\file_type.txt")
    #contactNameTypeList = getListForType("..\\calendar_peopleNames.20180119_used_for_some_patten.txt")

    #slotListDictionary = defaultdict(list);
    #samleOpenText(fileTypeList, "file_type".replace('_', '').upper() + 'HOLDER',  slotListDictionary)
    #samleOpenText(contactNameTypeList, "contact_name".replace('_', '').upper() + 'HOLDER',  slotListDictionary)

    slotListDictionary = defaultdict(list);
    for slot, slotHolderName in zip(slotList, HolderName):
        typeList = getListForType("..\\"+slot+".txt")
        samleOpenText(typeList, slot.replace('_', '').upper() + 'HOLDER', slotListDictionary)
    
    outputSet = [];
    for pattern in patternSet:

        # update seed 
        random.seed(0.1);


        
        array = pattern.split('\t');

        # verfiy how many columns they are
        if len(array) < 5:
            continue;
        
        originalQuery = array[1];
        intent = array[2];
        domain = array[3];
        originalSlotXml = array[4];

        for idx in range(hyper_parameter):
            query = originalQuery;
            slotXml = originalSlotXml;

            # key XXXHOLDER
            # value : something which should be inside <XMLTAG> </XMLTAG>
            for key, value in slotListDictionary.items():
                if key in query:
                    if key in MultipleHolderName:
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
                        inRangeIndex = random.randint(0, len(value)-1)

                        query = query.replace(key, value[inRangeIndex]);
                        slotXml = slotXml.replace(key, value[inRangeIndex]);
        
            outputSet.append('\t'.join(['0', query, intent, domain, slotXml]));

    outputSlot = '_'.join(slotList);
    
    if synthetic_mode == 1 or synthetic_mode == 5:
        # for slot
        print("generating slot synthetic...")
        with codecs.open('data_synthesised_' + outputSlot + '.tsv', 'w', 'utf-8') as fout:
            for item in outputSet:
                fout.write(item + '\r\n');

    if synthetic_mode == 2 or synthetic_mode == 5:
        # for intent
        print("generating intent synthetic...")
        with codecs.open('intent_data_synthesised_' + outputSlot + '.tsv', 'w', 'utf-8') as fout:
            for item in outputIntentSet:
                fout.write(item + '\r\n');   


if __name__ == '__main__':
    slotList = ['contact_name', 'file_keyword'];
    doubleSlotList = ['contact_name']
    #parse(slotList, hyper_parameter);
    parse(slotList, doubleSlotList);

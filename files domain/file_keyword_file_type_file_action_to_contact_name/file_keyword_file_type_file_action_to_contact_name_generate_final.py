import codecs;
import random;
from collections import defaultdict;

# 1: slot only
# 2: intent only
# 5: both
synthetic_mode = 1;


# add hyper paramter if unbalanced
sampling_hyper_paramter_each_slot = 200
# for intent using to prevent too many
#numberofQuery_hyper_parameter = 3
# here needs to generate data so numberofQuery_hyper_parameter == sampling_hyper_paramter_each_slot
numberofQuery_hyper_parameter = 200

# multiple holder randomization
multiple_holder_hyperparameter = 2

# random seed
# and inner loop random seed change
rand_seed_parameter_initialization = 0.1
rand_seed_offset = 0.01


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

def parse(slotList, removeSlotList, doubleSlotList):
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

        #if slot == 'sharetarget_type':
        #    print(typeList)
        samleOpenText(typeList, slot.replace('_', '').upper() + 'HOLDER', slotListDictionary)

    #initial rand seed	
    rand_seed_parameter = rand_seed_parameter_initialization
    
    outputSet = [];
    outputIntentSet = [];
    for pattern in patternSet:

	# update seed fpr each pattern	
        rand_seed_parameter=rand_seed_parameter+rand_seed_offset;	
        random.seed(rand_seed_parameter);


        
        array = pattern.split('\t');

        # verfiy how many columns they are
        if len(array) < 5:
            continue;
        
        originalQuery = array[1];
        intent = array[2];
        domain = array[3];
        originalSlotXml = array[4];

	# prevent duplication so use low bound of numberofQuery_hyper_parameter / sampling_hyper_paramter_each_slot	
        for idx in range(min(numberofQuery_hyper_parameter,sampling_hyper_paramter_each_slot)):
            query = originalQuery;
            slotXml = originalSlotXml;

            # key XXXHOLDER
            # value : something which should be inside <XMLTAG> </XMLTAG>
            for key, value in slotListDictionary.items():
                if key in query:
                    if key in MultipleHolderName:
                        # include
                        #cnt = random.randint(0, multiple_holder_hyperparameter)+1
                        cnt = random.randint(1, multiple_holder_hyperparameter)

                        #print(cnt)

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

            for removeSlot in removeSlotList:
                slotXml = slotXml.replace('<'+removeSlot+'>', "");
                slotXml = slotXml.replace('</'+removeSlot+'>', "");
        
            outputSet.append('\t'.join(['0', query, intent, domain, slotXml]));
            #TurnNumber	PreviousTurnIntent	query	intent
            outputIntentSet.append('\t'.join(['0',"\t"+query,intent]));

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
    slotList = ['file_keyword', 'file_type', 'to_contact_name'];
    removeSlotList = ['']

    # update lexcion to new so no need to double
    #doubleSlotList = ['to_contact_name']
    doubleSlotList = []
    
    parse(slotList, removeSlotList, doubleSlotList);

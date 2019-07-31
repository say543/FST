import codecs;
import random;
import re;

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
    "<data_source>" : "<file_location>",
    "</data_source>" : "</file_location>",
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
    "<order_ref> " : "",
    "</order_ref> " : "",
    "</order_ref>" : "",
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


fileTypeCandidate = []
fileNameCandidate = []
fileKeywordCandidate = []
fileContactNameCandidate = []
fileFromContactNameCandidate = []



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


            # fine-grained parse
            #list = re.findall("(</?[^>]*>)", slot)
            
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
                if xmlpair.startswith("<from_contact_name>"):
                    fileFromContactNameCandidate.append(xmlpair)

            

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

with codecs.open('files_mystuff_after_filtering_from_contact_name.tsv', 'w', 'utf-8') as fout:
    for item in fileFromContactNameCandidate:
        fout.write(item + '\r\n');




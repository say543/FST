import codecs;
import random;
import re;
import glob;
import os;


import math
import re
import sys

# add hyper paramter if unbalanced
hyper_parameter = 200



fileDomainRelatedIntent = ['file_search', 'file_open', 'file_share', 'file_download', 'file_other', 'file_navigate', "teamspace_search"]

teamsDomainToFileDomain = {
    # carina need small but bellevue needs big
    "teams" : "files",
    "TEAMS" : "FILES"
}



domainToNotSureDomain = {
    # carina need small but bellevue needs big
    "files" : "NOTSURE",
    "FILES" : "NOTSURE"
}



spellWrongButNotTags={
    'udocument ',
    ' udocument',    
    'focument ',
    ' focument',
    'documnt ',
    ' documnt',
    'docmuent ',
    ' docmuent',
    'documeny ',
    ' documeny',
    'doument ',
    ' doument',
    'documaent ',
    ' documaent',
    'documet ',
    ' documet',
    'documsnt ',
    ' documsnt'
    'documet ',
    ' documet',
    'documpent ',
    ' documpent',
    'docpument ',
    ' docpument',
    'docujent ',
    ' docujent',
    'bt ',
    ' bt',
    'cby ',
    ' cby',
    'ffiles ',
    ' ffiles',
    'fibnd ',
    ' fibnd',
    'fidn ',
    ' fidn',
    'fitled ',
    ' fitled',
    'tigled ',
    ' tigled',
    'titler ',
    ' titler',
    'titld ',
    ' titld',
    'sord ',
    ' sord',
    'spreadshset ',
    ' spreadshset',
    #'word y',
    ' word y',
    'bh ',
    ' bh',
    'ethe ',
    ' ethe',
    'mej ',
    ' mej',
    'methe ',
    ' methe',
    'findm ',
    ' findm',
    'wgord ',
    ' wgord',
    'fules ',
    ' fules',
    'dogc ',
    ' dogc',
    ' dcreate',
    'dcreate ',
    'fyiles ',
    ' fyiles',
    'dpcx ',
    ' dpcx',
    'fhe ',
    ' fhe',
    'docxv ',
    ' docxv',
    'youf ',
    ' youf',
    ' sharejd',
    'sharejd ',
    'word y ',
    'sharedl ',
    ' sharedl',
    'snared ',
    ' snared',
    ' iby',
    'iby ',
    'withy ',
    ' withy',
    'thd ',
    ' thd',
    'tby ',
    ' tby',
    'byy ',
    ' byy',
    ' namdd',
    'namdd ',
    'sbout ',
    ' sbout',
    'shqred ',
    ' shqred',
    'shraed ',
    ' shraed',
    'catlled ',
    ' catlled',
    'jamed ',
    ' jamed',
    'mdoified ',
    ' mdoified',
    'dovx ',
    ' dovx',
    'nmaed ',
    ' nmaed',
    'fule ',
    ' fule',
    'shred ',
    ' shred',
    'modiifed ',
    ' modiifed',
    'docxv ',
    ' docxv',
    'titped ',
    ' titped',
    'fimd ',
    ' fimd',
    'wotd ',
    ' wotd',
    'oepn ',
    ' oepn',
    'opeh ',
    ' opeh',
    'fkile ',
    ' fkile',
    'uby ',
    ' uby',
    'fike ',
    ' fike',
    'fyile ',
    ' fyile',
    'xldx ',
    ' xldx',
    'xhared ',
    ' xhared',
    'shard ',
    ' shard',
    'craeted ',
    ' craeted',
    'filr ',
    ' filr',
    'seht ',
    ' seht',
    'sharrd ',
    ' sharrd',
    'calledb ',
    ' calledb',
    'ifle ',
    ' ifle',
    'reated ',
    ' reated',
    'sharked ',
    ' sharked',
    'fjind ',
    ' fjind',

    # pay attention to only start and end
    # and queriy will do will be skipped
    ' do ',
    ' b ',
    ' m ',
    'show he ',
    ' fo ',
    ' rhe ',
    ' filec ',
    ' abot ',
    'documen ',
    ' yb ',
    ' ny ',
    ' hared',
    ' thr ',
    ' gy ',
    ' bout ',
    ' awith ',
    'exce by',
    ' oc ',
    'oc by',
    ' bu ',
    ' dthe ',
    ' je the ',
    ' lthe ',
    ' sthe ',
    ' documentb y ',
    ' ms ',
    'findthe ',
    'fle ',
    'ile ',
    'kpen ',
    ' Aprilb y ',
    'spreadsheetAshuthosh ',
    ' documentfiles ',
    'wore ',
    'sharedabout ',
    }

blackListQuerySet = {
    "go to my desktop",
}


fileTypeDomanBoost =set([
    'pptx',
    'ppts',
    'ppt',
    'deck',
    'decks',
    'presentation',
    'presentations',
    'powerpoint',
    'powerpoints',
    'power point',
    'slide',
    'slides',
    'doc',
    'docx',
    'docs',
    'spec',
    'excel',
    'excels',
    'xls',
    'xlsx',
    'spreadsheet',
    'spreadsheets',
    'workbook',
    'worksheet',
    'csv',
    'tsv',
    'note',
    'notes',
    'onenote',
    'onenotes',
    'onenote',
    'notebook',
    'notebooks',
    'pdf',
    'pdfs',
    'pdf',
    'jpg',
    'jpeg',
    'gif',
    'png',
    'image',
    'msg',
    'ics',
    'vcs',
    'vsdx',
    'vssx',
    'vstx',
    'vsdm',
    'vssm',
    'vstm',
    'vsd',
    'vdw',
    'vss',
    'vst',
    'mpp',
    'mpt',
    'word',
    'words',
    'document',
    'documents',
    'file',
    'files'
    ])

    
OutputSlotEvaluation = [];

OutputIntentEvaluation = [];

OutputSTCAIntentEvaluation = [];


# only files by judge annotation
OutputSpellFilterEvaluationOnlyFiles = [];
OutputSpellFilterEvaluation = [];

# only files by judge annotation, adding file_type filter to correct annotation
OutputSpellFilterEvaluationOnlyFilesDomainFileTypeFilter = [];
OutputSpellFilterEvaluationDomainFileTypeFilter = [];



OutputSpellWrongFilterEvaluation = [];






lexiconSet = set()
with codecs.open('..\\LexiconFiles\\lexicon.calendar.person_names_for_training.txt', 'r', 'utf-8') as fin:
    for line in fin:
        line = line.strip()
        lexiconSet.add(line)

filetypeSet = set()
with codecs.open('..\\lexicons\\file_type_domain_boost.txt', 'r', 'utf-8') as fin:
    for line in fin:
        line = line.strip()
        filetypeSet.add(line)


inputFile = "UHRS_Task_3SFilesAnnotation.Trimmed.Converted.tsv"

#with codecs.open('Teams-golden.tsv', 'r', 'utf-8') as fin:
with codecs.open(inputFile, 'r', 'utf-8') as fin:
    
    for line in fin:
        line = line.strip();
        if not line:
            continue;
        linestrs = line.split("\t");

        # make sure it at least has
        # Query	ExternalFeature	Weight	Intent	Domain	Slot
        #if len(linestrs) < 11:
        #    continue;

        # skip head
        if linestrs[5] == 'JudgedDomain':
            continue


        # replace all . with \t
                
        query = linestrs[4]

        # replace all  ./space/,/?/! with \t
        # not deal with PDF's
        # do it in the future
                
        query = str.replace(query, " ", "\t")
        query = str.replace(query, ".", "\t")
        query = str.replace(query, ",", "\t")
        query = str.replace(query, "?", "\t")
        query = str.replace(query, "!", "\t")

        querytrs = query.split("\t");

        hasFileType = False;
        for querystr in querytrs:
            if querystr.lower() in fileTypeDomanBoost:
                hasFileType = True
                break



        slot = linestrs[7]
        xmlpairs = re.findall("(<.*?>.*?<\/.*?>)", slot)

        invalid = False
        for xmlpair in xmlpairs:

            # extra type and value for xml tag
            xmlTypeEndInd = xmlpair.find(">")

            xmlType = xmlpair[1:xmlTypeEndInd]

            xmlValue = xmlpair.replace("<"+xmlType+">", "")
            xmlValue = xmlValue.replace("</"+xmlType+">", "")
            xmlValue = xmlValue.strip()

            xmlValueStrs = xmlValue.split(" ")

            if not invalid:
                if xmlType == 'file_type':
                    for xmlValueStr in xmlValueStrs:
                        if xmlValueStr.lower() not in filetypeSet:
                            #print(xmlValueStr.lower())
                            #OutputSpellWrongFilterEvaluation.append(line+"\t"+"file_type_error")
                            invalid = True
                            break

            if not invalid:
                if xmlType == 'contact_name' or xmlType == 'to_contact_name':            
                    for xmlValueStr in xmlValueStrs:
                        if xmlValueStr.lower() not in lexiconSet and xmlValueStr.lower() !="me" and xmlValueStr.lower() !="my":
                            #if xmlValueStr == "fVenkata":
                            #    print(xmlValueStr.lower())
                            #OutputSpellWrongFilterEvaluation.append(line+"\t"+"contact_name_error")
                            invalid = True
                            break

        if not invalid:
            for spellWrongButNotTag in spellWrongButNotTags:
                if linestrs[4].lower().find(spellWrongButNotTag.lower()) != -1:
                    #print("token:"+spellWrongButNotTag)
                    #print(linestrs[4])
                    invalid = True
                    break

        if not invalid and linestrs[4].lower().find("filetype=") !=-1 or linestrs[4].lower().find("filetype:")!=-1:
            #print(linestrs[4])
            invalid = True

            
                
        if invalid:
            #if linestrs[4] == 'word by Yaram fVenkata named 3S LiveSite':
            #    print('add')
            #print(line)
            OutputSpellWrongFilterEvaluation.append(line+"\t"+"contact_name_error")
        else:
            #print(line)
            #print(linestrs[5])
            OutputSpellFilterEvaluation.append(line)


            # in this case, modify doomains to NOTSURE so no need to add onlyFiles
            if not hasFileType and linestrs[5].lower() == 'files':
                #fout.write(linestrs[0]+'\t'+linestrs[1]+'\t'+linestrs[2]+'\t'+linestrs[3]+'\t'+linestrs[4]+'\t'+domainToFileDomainlinestrs[linestrs[5].lower()]+'\r\n');
                OutputSpellFilterEvaluationDomainFileTypeFilter.append(linestrs[0]+'\t'+linestrs[1]+'\t'+linestrs[2]+'\t'+linestrs[3]+'\t'+linestrs[4]+'\t'+domainToNotSureDomain[linestrs[5].lower()])
            else:
                if linestrs[5].lower() == 'files':
                    OutputSpellFilterEvaluationOnlyFilesDomainFileTypeFilter.append(linestrs[0]+'\t'+linestrs[1]+'\t'+linestrs[2]+'\t'+linestrs[3]+'\t'+linestrs[4]+'\t'+linestrs[5])
                OutputSpellFilterEvaluationDomainFileTypeFilter.append(linestrs[0]+'\t'+linestrs[1]+'\t'+linestrs[2]+'\t'+linestrs[3]+'\t'+linestrs[4]+'\t'+linestrs[5])
            
            if linestrs[5].lower() == 'files':
                OutputSpellFilterEvaluationOnlyFiles.append(line)

                # id / message / intent / domain / constraint
                # for training purpose's format
            
                #OutputSlotEvaluation.append("0"+"\t"+linestrs[4]+"\t"+linestrs[6]+"\t" +linestrs[5].lower()+"\t"+linestrs[7]);
                OutputSlotEvaluation.append("0"+"\t"+linestrs[4]+"\t"+linestrs[6]+"\t" +linestrs[5].lower()+"\t"+slot);

                # TurnNumber / PreviousTurnIntent / query /intent
                # for training purpose's format
                #OutputIntentEvaluation.append("0"+"\t"+""+"\t"+linestrs[4]+"\t" +linestrs[6]);
                OutputIntentEvaluation.append("0"+"\t"+""+"\t"+linestrs[4]+"\t" +linestrs[6]);

                #UUID\tQuery\tIntent\tDomain\tSlot\r\n
                #OutputSTCAIntentEvaluation.append("0"+"\t"+linestrs[4]+"\t" +linestrs[6]+"\t"+linestrs[5].lower()+"\t"+linestrs[7])
                OutputSTCAIntentEvaluation.append("0"+"\t"+linestrs[4]+"\t" +linestrs[6]+"\t"+linestrs[5].lower()+"\t"+slot)

"""
# comment shuffle in the first place
#random.shuffle(OutputSet);
"""

# for judge trainer format
#with codecs.open('teams_golden_after_filtering.tsv', 'w', 'utf-8') as fout:
#
#    # if outout originla format
#    fout.write("ConversationId\tMessageId\tMessageTimestamp\tMessageFrom\tMessageText\tJudgedDomain\tJudgedIntent\tJudgedConstraints\tMetaData\tConversationContext\tFrequency\tImplicitConstraints\r\n")
#    for item in Output:
#        fout.write(item + '\r\n');

# for CMF slot evaluation format
with codecs.open((inputFile.split("."))[0] +'slot_evaluation.tsv', 'w', 'utf-8') as fout:

    # if output for traing
    fout.write("id\tquery\tintent\tdomain\tQueryXml\r\n")
    for item in OutputSlotEvaluation:
        fout.write(item + '\r\n');

# for STCA evaluation
with codecs.open("sharemodeltest\\"+(inputFile.split("."))[0] +'slot_evaluation.tsv', 'w', 'utf-8') as fout:

    # if output for traing
    fout.write("id\tquery\tintent\tdomain\tQueryXml\r\n")
    for item in OutputSlotEvaluation:
        fout.write(item + '\r\n');

# for CMF intent evaluation format
with codecs.open((inputFile.split("."))[0] +'intent_evaluation.tsv', 'w', 'utf-8') as fout:

    # if output for traing
    fout.write("TurnNumber\tPreviousTurnIntent\tquery\tintent\r\n")
    for item in OutputIntentEvaluation:
        fout.write(item + '\r\n');

# for STCAevaluation 
with codecs.open("sharemodeltest\\"+(inputFile.split("."))[0] +'intent_evaluation.tsv', 'w', 'utf-8') as fout:

    # if output for traing
    fout.write("UUID\tQuery\tIntent\tDomain\tSlot\r\n")
    for item in OutputSTCAIntentEvaluation:
        fout.write(item + '\r\n');

# for spelling checking
# spell correct for Bellevue evaluation
# for spelling checking
with codecs.open((inputFile.split("."))[0] +'_after_filtering.tsv', 'w', 'utf-8') as fout:

    # if output for traing
    fout.write("ConversationId\tMessageId\tMessageTimestamp\tMessageFrom\tMessageText\tJudgedDomain\tJudgedIntent\tJudgedConstraints\tMetaData\tConversationContext\tFrequency\tCortanaResponse\r\n")
    for item in OutputSpellFilterEvaluationOnlyFiles:
        fout.write(item + '\r\n');

with codecs.open((inputFile.split("."))[0] +'_after_filtering_all.tsv', 'w', 'utf-8') as fout:
    # if output for traing
    fout.write("ConversationId\tMessageId\tMessageTimestamp\tMessageFrom\tMessageText\tJudgedDomain\tJudgedIntent\tJudgedConstraints\tMetaData\tConversationContext\tFrequency\tCortanaResponse\r\n")
    for item in OutputSpellFilterEvaluation:
        fout.write(item + '\r\n');




with codecs.open((inputFile.split("."))[0] +'_after_filtering_domain_with_file_type_correction.tsv', 'w', 'utf-8') as fout:

    # if output for traing
    fout.write("ConversationId\tMessageId\tMessageTimestamp\tMessageFrom\tMessageText\tJudgedDomain\r\n")
    for item in OutputSpellFilterEvaluationOnlyFilesDomainFileTypeFilter:
        fout.write(item + '\r\n');

with codecs.open((inputFile.split("."))[0] +'_after_filtering_all_domain_with_file_type_correction.tsv', 'w', 'utf-8') as fout:
    # if output for traing
    fout.write("ConversationId\tMessageId\tMessageTimestamp\tMessageFrom\tMessageText\tJudgedDomain\r\n")
    for item in OutputSpellFilterEvaluationDomainFileTypeFilter:
        fout.write(item + '\r\n');



        



with codecs.open((inputFile.split("."))[0] +'_spell_wrong.tsv', 'w', 'utf-8') as fout:

    # if output for traing
    fout.write("ConversationId\tMessageId\tMessageTimestamp\tMessageFrom\tMessageText\tJudgedDomain\tJudgedIntent\tJudgedConstraints\tMetaData\tConversationContext\tFrequency\tCortanaResponse\tErrorSource\r\n")
    for item in OutputSpellWrongFilterEvaluation:
        fout.write(item + '\r\n');

#######################
# intent level output
#######################
'''
with codecs.open('teams_slot_training_after_filtering_teamspace_search.tsv', 'w', 'utf-8') as fout:
    for item in teamspaceSearchCandidateSet:
        fout.write(item + '\r\n');
'''


#######################
# slot level output
#######################

'''
with codecs.open('teams_slot_training_after_filtering_file_keyword.tsv', 'w', 'utf-8') as fout:
    for item in fileKeywordCandidateSet:
        fout.write(item + '\r\n');

with codecs.open('teams_slot_training_after_filtering_file_name.tsv', 'w', 'utf-8') as fout:
    for item in fileNameCandidateSet:
        fout.write(item + '\r\n');

with codecs.open('teams_slot_training_after_filtering_meeting_starttime.tsv', 'w', 'utf-8') as fout:
    for item in meetingStarttimeCandidateSet:
        fout.write(item + '\r\n');

with codecs.open('teams_slot_training_after_filtering_file_type.tsv', 'w', 'utf-8') as fout:
    for item in fileTypeCandidateSet:
        fout.write(item + '\r\n');

# this is to deduplication
with codecs.open('teams_slot_training_after_filtering_file_recency.tsv', 'w', 'utf-8') as fout:
    for item in fileRecencyCandidateSet:
        fout.write(item + '\r\n');

with codecs.open('teams_slot_training_after_filtering_sharetarget_type.tsv', 'w', 'utf-8') as fout:
    for item in sharetargetTypeCandidateSet:
        fout.write(item + '\r\n');

with codecs.open('teams_slot_training_after_filtering_sharetarget_name.tsv', 'w', 'utf-8') as fout:
    for item in sharetargetNameCandidateSet:
        fout.write(item + '\r\n');

with codecs.open('teams_slot_training_after_filtering_contact_name.tsv', 'w', 'utf-8') as fout:
    for item in contactNameCandidateSet:
        fout.write(item + '\r\n');

with codecs.open('teams_slot_training_after_filtering_file_action.tsv', 'w', 'utf-8') as fout:
    for item in fileActionCandidateSet:
        fout.write(item + '\r\n');
        
with codecs.open('teams_slot_training_after_filtering_order_ref.tsv', 'w', 'utf-8') as fout:
    for item in orderRefCandidateSet:
        fout.write(item + '\r\n');
'''

#######################
# query replacement revert
#######################

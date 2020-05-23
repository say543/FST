import codecs;
import random;
import re;
import glob;
import os;


import math
import re
import sys

# add hyper paramter if unbalanced
sampling_hyper_paramter_each_cat = 50

#test
#repeat_times = 200

extra_amount_list = [1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000]


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
    'excel',
    'excels',
    'xls',
    'xlsx',
    'sheet',
    'sheets',
    'spreadsheet',
    'spreadsheets',
    'workbook',
    'worksheet',
    'csv',
    'tsv',
    'onenote',
    'onenotes',
    'onenote',
    'pdf',
    'pdfs',
    'pdf',
    'png',
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



def extractBybound(catset, holderList):
    # read open text content

    # 
    bound = min(len(catset),sampling_hyper_paramter_each_cat)

    cnt = 0
    for item in catset:
        if cnt >= bound:
            break
        holderList.append(item)
        cnt+=1
    
    return holderList

    
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



OutputDomainEvaluation = [];
OutputDomainFilter = [];


lexiconSet = set()
with codecs.open('..\\LexiconFiles\\lexicon.calendar.person_names_for_training.txt', 'r', 'utf-8') as fin:
    for line in fin:
        line = line.strip()
        lexiconSet.add(line)





# difference
# extra in normal boost
# spec will be removed right know leave it becasue we want to enhance possible file type in the future
#spec

# four kinds using pattern to cover, not trainnig  since conflicting with ondevice more
#note	
#notes
#notebook	
#notebooks

# eventually  all clients will not support those
#jpg	
#jpeg	
#gif
#image

# so using uwp one

filetypeSet = set()
with codecs.open('..\\lexicons\\file_type_domain_boost_UWP.txt', 'r', 'utf-8') as fin:
    for line in fin:
        line = line.strip()
        filetypeSet.add(line)


inputFile = "domain_extract_from_slot_query.tsv"

#with codecs.open('Teams-golden.tsv', 'r', 'utf-8') as fin:
with codecs.open(inputFile, 'r', 'utf-8') as fin:
    
    for line in fin:
        line = line.strip();
        if not line:
            continue;
        linestrs = line.split("\t");


        # replace all . with \t
                
        query = linestrs[0]

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

    


        if hasFileType:
            OutputDomainEvaluation.append("0"+"\t\t"+linestrs[0]+"\t"+"files"+"\t\t\t\t\t\t\t"+inputFile);
        else:
            OutputDomainFilter.append(line)


            
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



downloadCnt = 0
downloadmyCnt = 0
downloadthisCnt = 0
downloadtheCnt = 0
downloadCnt = 0
downloadsCnt = 0
downloadedCnt = 0

opentheCnt = 0
openupCnt = 0
openmyCnt = 0
openCnt = 0


sendtheCnt = 0
sendCnt = 0


sharetheCnt = 0
sharemyCnt = 0
shareCnt = 0


showtheCnt = 0
showmeCnt = 0
showmyCnt = 0
showCnt = 0


downloadSet =set([])
downloadmySet =set([])
downloadthisSet =set([])
downloadtheSet =set([])
downloadsSet =set([])
downloadedSet =set([])

opentheSet =set([])
openupSet =set([])
openmySet =set([])
openSet =set([])


sendtheSet =set([])
sendSet =set([])


sharetheSet =set([])
sharemySet =set([])
shareSet =set([])


showtheSet =set([])
showmeSet =set([])
showmySet =set([])
showSet =set([])


catSetlist = []

catSetlist.append(downloadSet)
catSetlist.append(downloadmySet)
catSetlist.append(downloadthisSet)
catSetlist.append(downloadtheSet)
catSetlist.append(downloadsSet)
catSetlist.append(downloadedSet)

catSetlist.append(opentheSet)
catSetlist.append(openupSet)
catSetlist.append(openmySet)
catSetlist.append(openSet)

catSetlist.append(sendtheSet)
catSetlist.append(sendSet)

catSetlist.append(sharetheSet)
catSetlist.append(sharemySet)
catSetlist.append(shareSet)

catSetlist.append(showtheSet)
catSetlist.append(showmeSet)
catSetlist.append(showmySet)
catSetlist.append(showSet)





for item in OutputDomainEvaluation:

    linestrs = item.split("\t");

    # for debug
    #print (linestrs[2])

    # extra space to male sure ending
    # order does matter
    
    if (linestrs[2].lower().startswith('downloads ')):
        downloadsCnt+=1
        downloadsSet.add(item)
    elif (linestrs[2].lower().startswith('download my ')):
        downloadmyCnt+=1
        downloadmySet.add(item)
    elif (linestrs[2].lower().startswith('download this ')):
        downloadthisCnt+=1
        downloadthisSet.add(item)
    elif (linestrs[2].lower().startswith('download the ')):
        downloadtheCnt+=1
        downloadtheSet.add(item)        
    elif (linestrs[2].lower().startswith('download ')):
        downloadCnt+=1
        downloadSet.add(item)
    elif (linestrs[2].lower().startswith('downloaded ')):
        downloadedCnt+=1
        downloadedSet.add(item)
    elif (linestrs[2].lower().startswith('open the ')):
        opentheCnt+=1
        opentheSet.add(item)
    elif (linestrs[2].lower().startswith('open up ')):
        openupCnt+=1
        openupSet.add(item)
    elif (linestrs[2].lower().startswith('open my ')):
        openmyCnt+=1
        openmySet.add(item)
    elif (linestrs[2].lower().startswith('open ')):
        openCnt+=1
        openSet.add(item)
    elif (linestrs[2].lower().startswith('send the')):
        sendtheCnt+=1
        sendtheSet.add(item)
    elif (linestrs[2].lower().startswith('send ')):
        sendCnt+=1
        sendSet.add(item)
    elif (linestrs[2].lower().startswith('share the ')):
        sharetheCnt+=1
        sharetheSet.add(item)
    elif (linestrs[2].lower().startswith('share my ')):
        sharemyCnt+=1
        sharemySet.add(item)
    elif (linestrs[2].lower().startswith('share ')):
        shareCnt+=1
        shareSet.add(item)
    elif (linestrs[2].lower().startswith('show the ')):
        showtheCnt+=1
        showtheSet.add(item)
    elif (linestrs[2].lower().startswith('show me ')):
        showmeCnt+=1
        showmeSet.add(item)
    elif (linestrs[2].lower().startswith('show my ')):
        showmyCnt+=1
        showmySet.add(item)
    elif (linestrs[2].lower().startswith('show ')):
        showCnt+=1
        showSet.add(item)
        
    '''
    if (linestrs[2].lower().startswith('open ')):
        openCnt+=1
        openSet.add(item)
    if (linestrs[2].lower().startswith('send ')):
        sendCnt+=1
        sendSet.add(item)
    if (linestrs[2].lower().startswith('share ')):
        shareCnt+=1
        shareSet.add(item)
    if (linestrs[2].lower().startswith('show ')):
        showCnt+=1
        showSet.add(item)
    '''

#print('downloadCnt:' + str(downloadCnt))
#print('openCnt:' + str(openCnt))
#print('sendCnt:' + str(sendCnt))
#print('shareCnt:' + str(shareCnt))
#print('showCnt:' + str(showCnt))

print('dedup downloadCnt:' + str(len(downloadSet)))
print('dedup downloadmySet:' + str(len(downloadmySet)))
print('dedup downloadthisSet:' + str(len(downloadthisSet)))
print('dedup downloadtheSet:' + str(len(downloadtheSet)))
print('dedup downloadsSet:' + str(len(downloadsSet)))
print('dedup downloadedSet:' + str(len(downloadedSet)))

print('dedup opentheSet:' + str(len(opentheSet)))
print('dedup openupSet:' + str(len(openupSet)))
print('dedup openmySet:' + str(len(openmySet)))
print('dedup openCnt:' + str(len(openSet)))



print('dedup sendtheSet:' + str(len(sendtheSet)))
print('dedup sendCnt:' + str(len(sendSet)))


print('dedup sharetheSet:' + str(len(sharetheSet)))
print('dedup sharemySet:' + str(len(sharemySet)))
print('dedup shareCnt:' + str(len(shareSet)))


print('dedup showtheSet:' + str(len(showtheSet)))
print('dedup showmeSet:' + str(len(showmeSet)))
print('dedup showmySet:' + str(len(showmySet)))
print('dedup showCnt:' + str(len(showSet)))


# for CMF slot evaluation format
# using undedup data
#with codecs.open((inputFile.split("."))[0] +'_domain_extraction.tsv', 'w', 'utf-8') as fout:

    # if output for traing
#    fout.write("TurnNumber\tPreviousTurnIntent\tquery\tdomain\tPreviousTurnDomain\tTaskFrameStatus\tTaskFrameEntityStates\tTaskFrameGUID\tSpeechPeopleDisambiguationGrammarMatches\tConversationalContext\tSource\r\n")
    
#    for item in OutputDomainEvaluation:
#        fout.write(item + '\r\n');




for extra_amount in extra_amount_list:
    with codecs.open((inputFile.split("."))[0] +'_domain_extraction.tsv', 'w', 'utf-8') as fout, codecs.open((inputFile.split("."))[0] +'_domain_extraction_'+str(extra_amount)+'.tsv', 'w', 'utf-8') as fout2:

        # if output for traing
        fout.write("TurnNumber\tPreviousTurnIntent\tquery\tdomain\tPreviousTurnDomain\tTaskFrameStatus\tTaskFrameEntityStates\tTaskFrameGUID\tSpeechPeopleDisambiguationGrammarMatches\tConversationalContext\tSource\r\n")
        fout2.write("TurnNumber\tPreviousTurnIntent\tquery\tdomain\tPreviousTurnDomain\tTaskFrameStatus\tTaskFrameEntityStates\tTaskFrameGUID\tSpeechPeopleDisambiguationGrammarMatches\tConversationalContext\tSource\r\n")

        # each set select extra_amount
        for eleset in catSetlist:
            cnt = 0 
            while cnt < extra_amount:
                for item in eleset:
                    fout.write(item + '\r\n');
                    fout2.write(item + '\r\n');
                    cnt+=1
                    if cnt >= extra_amount:
                        break

'''
# usiong dedup data
with codecs.open((inputFile.split("."))[0] +'_domain_extraction.tsv', 'w', 'utf-8') as fout:

    # if output for traing
    fout.write("TurnNumber\tPreviousTurnIntent\tquery\tdomain\tPreviousTurnDomain\tTaskFrameStatus\tTaskFrameEntityStates\tTaskFrameGUID\tSpeechPeopleDisambiguationGrammarMatches\tConversationalContext\tSource\r\n")

    
    # no shuffle right now
    # extra sampling_hyper_paramter_each_cat
    # if less than that, extra all
    #holderList = []
    #for eleset in catSetlist:
    #    holderList = extractBybound(eleset, holderList)
     
    #for item in holderList:
    #    fout.write(item + '\r\n');


    # for unique queries
    # testing repeat times
    for eleset in catSetlist:
        for item in eleset:


            

            # repeat = 100
            # 2400000
            # test repeat time for each query
            for index in range(repeat_times):
                fout.write(item + '\r\n');
    

    # using originla data
    
    #for item in downloadSet:
    #    fout.write(item + '\r\n');
    #for item in openSet:
    #    fout.write(item + '\r\n');
    #for item in sendSet:
    #    fout.write(item + '\r\n');
    #for item in shareSet:
    #    fout.write(item + '\r\n');
    #for item in showSet:
    #    fout.write(item + '\r\n');
   
'''



with codecs.open((inputFile.split("."))[0] +'_filtered.tsv', 'w', 'utf-8') as fout:

    # no head to for debug only
    #fout.write("TurnNumber\tPreviousTurnIntent\tquery\tdomain\tPreviousTurnDomain\tTaskFrameStatus\tTaskFrameEntityStates\tTaskFrameGUID\tSpeechPeopleDisambiguationGrammarMatches\tConversationalContext\tSource\r\n")
    
    for item in OutputDomainFilter:
        fout.write(item + '\r\n');

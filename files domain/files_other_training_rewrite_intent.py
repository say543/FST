import codecs;
import random;

# add hyper paramter if unbalanced
hyper_parameter = 200

PREVIOUSTURNDOMAIN = "PreviousTurnDomain"
PREVIOUSTURNINTENT = "PreviousTurnIntent"
TASKFRAMESTATUS = "TaskFrameStatus"
TASKFRAMEENTITYSTATES = "TaskFrameEntityStates"
TASKFRAMEGUID = "TaskFrameGUID"
SPEECHPEOPLEDISAMBIGUATIONGRAMMARMATCHES = "SpeechPeopleDisambiguationGrammarMatches"
CONVERSATIONALCONTEXT = "ConversationalContext"



commonDomainRelatedIntent = ['cancel','confirm','reject','select_none']

rewriteIntent = "file_other" 

OutputSet = [];
with codecs.open('files_other_training.tsv', 'r', 'utf-8') as fin:
    for line in fin:
        line = line.strip();
        if not line:
            continue;
        linestrs = line.split("\t")
        if len(linestrs) < 3:
            continue;

        #skip common domain intent
        #if linestrs[3] not in commonDomainRelatedIntent:
        #    OutputSet.append(linestrs[0]+"\t\t"+linestrs[2]+"\t"+rewriteIntent);

        # add
        # "PreviousTurnDomain"
        # "PreviousTurnIntent"
        # as
        # 'TurnNumber', PREVIOUSTURNINTENT, 'query', 'intent',PREVIOUSTURNDOMAIN])
        # append empty at first
        if linestrs[3] not in commonDomainRelatedIntent:
            OutputSet.append(linestrs[0]+"\t\t"+linestrs[2]+"\t"+rewriteIntent+"\t");


# commna shuffle at first
#random.shuffle(OutputSet);

with codecs.open('files_other_training_after_rewrite.tsv', 'w', 'utf-8') as fout:
    fout.write('\t'.join(['TurnNumber', PREVIOUSTURNINTENT, 'query', 'intent',PREVIOUSTURNDOMAIN, TASKFRAMESTATUS, TASKFRAMEENTITYSTATES, TASKFRAMEGUID, SPEECHPEOPLEDISAMBIGUATIONGRAMMARMATCHES, CONVERSATIONALCONTEXT])+'\r\n');
    for item in OutputSet:
        fout.write(item + '\r\n');
    


        

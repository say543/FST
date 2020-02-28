import codecs;
import random;

# add hyper paramter if unbalanced
hyper_parameter = 200

# add repeated times
repated_time = 5

# 07132019 add file_navigate
#fileDomainRelatedIntent = ['file_search', 'file_open', 'file_share', 'file_download', 'file_other']
# no need to add file nvatigate for intent training
# only slot is needed
# add teamspace_search but filter with different rules 
#fileDomainRelatedIntent = ['file_search', 'file_open', 'file_share', 'file_download', 'file_other', 'file_navigate']
fileDomainRelatedIntent = ['file_search', 'file_open', 'file_share', 'file_download', 'file_other', 'file_navigate', 'teamspace_search']


##############################
# intent level candidate
##############################
teamspaceSearchCandidateSet = set()


Output = [];

updateQuery = set()

prefixWtihVerbWithMy =set([
    #empty is also fine
    "",
    "mystuff ",
    "Can you ",
    "can you ",
    "Can ",
    "could you ",
    "would you ",
    "Please ",
    "please ",
    "Can you please ",
    "can you please ",
    "would you please ",
    "hey cortana please ",
    "Hey cortana please ",
    "cortana please ",
    "cortana ",
    "cortana to",
    "Hey ",
    "hey cortana ",
    "Hey cortana can you ",
    "hey cortana can you ",
    "hey cortana would you ",
    "Hey can you ",
    "Hey can you ",
    "Hey could you ",
    "Hey would you ",
    "i want to ",
    "I want to ",
    "i need to ",
    "I need to ",
    "i have to ",
    "I have to ",
    "i required to ",
    "i want to ",
    "i want you to ",
    "I want you to ",
    "i can't ",
    "I can't ",
    "I would like to ",
    "I would love to ",
    "I'd like to ",
    "I will ",
    "Try to ",
    "try to ",
    "Cortana, ",
    "cortana, ",
    "Cortana, please ",
    "cortana, please ",
    "Cortana, help me ",
    "cortana, help me ",
    "cortana, i need to ",
    "Cortana, i need to ",
    "cortana can you ",
    "Cortana can you "
    "cortana could you ",
    "cortana would you ",
    "hi ",
    "hi what's up ",
    "Hi what's up ",
    "Hi please ",
    "Hello please ",
    "hello can you ",
    "Hello would you ",
    "Hello could you ",
    "can you help me ",
    "Can you help me ",
    "could you help me ",
    "would you help me ",
    "can you help ",
    "Can you help ",
    "could you help ",
    "would you help ",
    "help me ",
    "Help me ",
    "where do i ",
    "Where do i ",
    "where do I ",
    "Where do I ",
    "where will i ",
    "Where will i ",
    "where will I ",
    "Where will I ",
    "where can i ",
    "Where can i ",
    "where can I ",
    "Where can I ",
    "where do i ",
    "Where do i ",
    "where do I ",
    "Where do I ",
    "Siri ",
    "siri ",
    "Bing ",
    "bing ",
    #empty is also fine
    "",
    ])

with codecs.open('teams_intent_training.tsv', 'r', 'utf-8') as fin:
    for line in fin:
        line = line.strip();
        if not line:
            continue;
        linestrs = line.split("\t");
        if len(linestrs) < 4:
            continue;
        
        if linestrs[3] in fileDomainRelatedIntent:

            # if file_other
            # ignore
            if linestrs[3] == 'file_other':
                print("other intent")
                print(linestrs[3])
                print(linestrs[2])

            # if file_navigate

            # document teamspace_search
            # for further analysis
            # skip it at first

            teamspaceSearchMatch = False
            if linestrs[3] == "teamspace_search":
                # having problem or tab
                # then skip
                if linestrs[2].find("program") != -1 or \
                   linestrs[2].find("tab") != -1 or \
                   linestrs[2].find("channel") != -1 or \
                   linestrs[2].find("team") != -1 or \
                   linestrs[2].find("teams") != -1 or \
                   linestrs[2].find("conversation") != -1 or \
                   linestrs[2].find("chat") != -1:
                    teamspaceSearchCandidateSet.add(line)
                    continue
                linestrs[3] = "file_search"
                teamspaceSearchMatch = True

            # skip teamspace_search query
            if teamspaceSearchMatch:
                continue
                

            naviverbs = set([
                        "go to ",
                        "going to ",
                        "return to ",
                        "returning to ",
                        "gogo to ",
                        "navigate ",
                        "navigating ",
                        "navigate to ",
                        "navigating to ",
                        "navigate with ",
                        "navigating with ",                        
                        "navigate for ",
                        "navigating for ",
                        "navigate in ",
                        "navigating in ",
                        "navigate is ",
                        "navigating is ",
                        "navigate of ",
                        "navigating of ",
                        # remove below section and move to file_search
                        #"bring up ",
                        #"bringing up ",
                        #"pull up ",
                        #"pull of ",
                        #"pulling of ",
                        #"pulls up ",
                        #"put up ",
                        #"pulling up ",
                        #"pulling ",
                        #"bring ",
                        #"bringing ",
                        #"bring back up ",
                        #"bringing back up ",
                        #"brought up ",
                        #"bring of ",
                        #"bringing of ",
                        #"bring to ",
                        #"bringing to ",
                        # remove above section and move to file_search
                        "browse ",
                        "browsing ",
                        "see ",
                        "seeing ",
                        "visit ",
                        "visitng ",
                        "raise ",
                        "raising ",
                        # send is file_share
                        #"send ",
                        #"sending ",
                        "make up ",
                        "making up ",
                        # remove below section and move to file_search
                        #"take ",
                        #"taking ",
                        #"take me ",
                        #"taking me ",
                        #"took me ",
                        #"take me to ",
                        #"taking me to ",
                        #"take me out to ",
                        #"taking me out to ",
                        #"take up ",
                        #"taking up ",
                        # remove above section and move to file_search
                        "give ",
                        "giving ",
                        "go back to ",
                        "going back to ",
                        "go for ",
                        "going for ",
                        "go in ",
                        "going in ",
                        "go is ",
                        "going is ",
                        "go of ",
                        "going of ",
                        "go ",
                        "going ",
                        "direct ",
                        "directing ",
                        "list "
                        "listing "
                        "what is ",
                        "what was ",
                        "what are ",
                        "what was ",
                        "what were ",
                        ])

            openverbs = set([
                        "open ",
                        "opening ",
                        "view ",
                        "viewing ",
                        ])

            # add ing as verb
            searchverbs = set(["search ",
                        "searched ",
                        "find ",
                        "found ",
                        "look for ",
                        "looked for ",
                        "look up ",
                        "looked up ",
                        "show ",
                        "showed ",
                        "display ",
                        "displayed ",
                        # below section and move to file_search
                        "bring up ",
                        "bringing up ",
                        "pull up ",
                        "pull of ",
                        "pulling of ",
                        "pulls up ",
                        "put up ",
                        "pulling up ",
                        "pulling ",
                        "bring ",
                        "bringing ",
                        "bring back up ",
                        "bringing back up ",
                        "brought up ",
                        "bring of ",
                        "bringing of ",
                        "bring to ",
                        "bringing to ",
                        # above section and move to file_search
                        # below section and move to file_search
                        "take ",
                        "taking ",
                        "take me ",
                        "taking me ",
                        "took me ",
                        "take me to ",
                        "taking me to ",
                        "take me out to ",
                        "taking me out to ",
                        "take up ",
                        "taking up ",
                        # above section and move to file_search
                        "locate ",
                        "locate ",   
                        ])

            # add first one , second one.....
            # with number inside
            downloadverbs = set(["download ",
                        "downloading ",
                        "save ",
                        "saving "
                        "upload ",
                        "uploading "
                        "keep ",
                        "keeping "                                 
                        ])

            shareverbs = set(["share ",
                        ])

            #print(linestrs[1])

            intent = linestrs[3]

            # also file_open
            #if linestrs[3] == "file_search" or linestrs[3] == "file_navigate":
            if linestrs[3] == "file_search" or linestrs[3] == "file_navigate" or linestrs[3] == "file_open":

                # start
                match = False

                if not match:
                    for prefix in prefixWtihVerbWithMy:
                        for verb in naviverbs:
                            if linestrs[2].lower().startswith((prefix+verb).lower()):
                                intent = "file_navigate"
                                match = True
                                break
                if not match:
                    for prefix in prefixWtihVerbWithMy:
                        for verb in openverbs:
                            if linestrs[2].lower().startswith((prefix+verb).lower()):
                                if ((verb.lower().startswith('open')) and linestrs[2].lower().find("with") != -1):
                                    print(linestrs[2])
                                    intent = "file_share"
                                else:
                                    #print(linestrs[2])
                                    intent = "file_open"
                                match = True
                                break

                if not match:
                    for prefix in prefixWtihVerbWithMy:
                        for verb in searchverbs:
                            if linestrs[2].lower().startswith((prefix+verb).lower()):
                                if ((verb.lower().startswith('show') or verb.lower().startswith('shell')) and linestrs[2].lower().find("with") != -1):
                                    print(linestrs[2])
                                    intent = "file_share"
                                else:
                                    #print(linestrs[2])
                                    intent = "file_search"
                                match = True
                                break

                if not match:
                    for prefix in prefixWtihVerbWithMy:
                        for verb in naviverbs:
                            if linestrs[2].lower().find((prefix+verb).lower()) != -1:
                                intent = "file_navigate"
                                match = True
                                break
                if not match:
                    for prefix in prefixWtihVerbWithMy:
                        for verb in openverbs:
                            if linestrs[2].lower().find((prefix+verb).lower()) != -1:
                                if ((verb.lower().startswith('open')) and linestrs[2].lower().find("with") != -1):
                                    print(linestrs[2])
                                    intent = "file_share"
                                else:
                                    #print(linestrs[2])
                                    intent = "file_open"
                                match = True
                                break

                if not match:
                    for prefix in prefixWtihVerbWithMy:
                        for verb in searchverbs:
                            if linestrs[2].lower().find((prefix+verb).lower()) != -1:
                                if ((verb.lower().startswith('show') or verb.lower().startswith('shell')) and linestrs[2].lower().find("with") != -1):
                                    print(linestrs[2])
                                    intent = "file_share"
                                else:
                                    #print(linestrs[2])
                                    intent = "file_search"
                                match = True
                                break

                        
                

            '''
            match = False

            # each one should be exclusive
            intent = 'file_search'
            for prefix in prefixWtihVerbWithMy:
                for verb in naviverbs:
                    if linestrs[2].lower().startswith((prefix+verb).lower()):
                        intent = "file_navigate"
                        match = True
                        break

            for prefix in prefixWtihVerbWithMy:
                for verb in openverbs:
                    if linestrs[2].lower().startswith((prefix+verb).lower()):
                        intent = "file_open"
                        match = True
                        break
        
            #for prefix in prefixWtihVerbWithMy:
            #    for verb in searchverbs:
            #        if linestrs[2].lower().startswith((prefix+verb).lower()):
            #            intent = "file_search"
            #            match = True
            #            break

            for prefix in prefixWtihVerbWithMy:
                for verb in downloadverbs:
                    if linestrs[2].lower().startswith((prefix+verb).lower()):
                        intent = "file_download"
                        match = True
                        break

            for prefix in prefixWtihVerbWithMy:
                for verb in shareverbs:
                    if linestrs[2].lower().startswith((prefix+verb).lower()):
                        intent = "file_share"
                        match = True
                        break
            '''

            # teams has more kinds of intent so do not use file_search as intent                   
            
            if intent != linestrs[3]:

                updateQuery.add(linestrs[2]+"\t"+linestrs[3] + "\t" + intent)
                #print(linestrs[2])
                #print("orig:"+linestrs[3])
                #print("new:"+intent)
            linestrs[3] = intent
                

            

            for i in range(0,repated_time):
                # TurnNumber	PreviousTurnIntent	query	intent
                Output.append(linestrs[0]+"\t"+linestrs[1]+"\t"+linestrs[2]+"\t"+linestrs[3]);

"""
# comment shuffle in the first place
#random.shuffle(OutputSet);
"""


#######################
# intent level output
#######################
with codecs.open('teams_intent_training_after_filtering.tsv', 'w', 'utf-8') as fout:
    for item in Output:
        fout.write(item + '\r\n');


with codecs.open('teams_intent_update_after_filtering.tsv', 'w', 'utf-8') as fout:
    for item in updateQuery:
        fout.write(item + '\r\n');

        
with codecs.open('teams_intent_training_after_filtering_teamspace_search.tsv', 'w', 'utf-8') as fout:
    for item in teamspaceSearchCandidateSet:
        fout.write(item + '\r\n');

        

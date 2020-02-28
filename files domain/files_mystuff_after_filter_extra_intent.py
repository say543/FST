import codecs;
import random;

# add hyper paramter if unbalanced
hyper_parameter = 200

# add repeated times
#my stuff data too much
# so repeat one time
repated_time = 1


# upper_cnt
# to match with teams intent total data
upper_cnt = 7000

fileDomainRelatedIntent = ['file_search', 'file_open', 'file_share', 'file_download', 'file_other']

Output = [];

updateQuery = set()

# here low case or bigger case does not matter since only for intent checking
prefixWtihVerbWithMy =set([
    #empty is also fine
    "",
    "mystuff ",
    "Can you ",
    "can you ",
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
    "Hey cortana ",
    "hey cortana ",
    "Hey cortana can you ",
    "hey cortana can you ",
    "hey cortana would you ",
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
    "I'd like to",
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

with codecs.open('files_mystuff_after_filtering.tsv', 'r', 'utf-8') as fin:
    for line in fin:
        line = line.strip();
        if not line:
            continue;
        linestrs = line.split("\t")
        if len(linestrs) < 5:
            continue;
        

        if linestrs[2] in fileDomainRelatedIntent:

            # following the guideline to ove queries to file_navigate intent
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
                        "bring up ",
                        "bringing up ",
                        "pull up ",
                        "pull of ",
                        "pulling of ",
                        "pulls up ",
                        "put up ",
                        "pulling up ",
                        "pulling ",
                        "raise ",
                        "raising ",
                        "visit ",
                        "visitng ",
                        "bring ",
                        "bringing ",
                        "bring back up ",
                        "bringing back up ",
                        "brought up ",
                        "bring of ",
                        "bringing of ",
                        "bring to ",
                        "bringing to ",
                        "browse ",
                        "browsing ",
                        "see ",
                        "seeing ",
                        #"send ",
                        #"sending ",
                        "make up ",
                        "making up ",
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

            intent = linestrs[2]
            match = False
            if not match:
                for prefix in prefixWtihVerbWithMy:
                    for verb in naviverbs:
                        if linestrs[1].lower().startswith((prefix+verb).lower()):
                            intent = "file_navigate"
                            match = True
                            break
            if not match:
                for prefix in prefixWtihVerbWithMy:
                    for verb in openverbs:
                        if linestrs[1].lower().startswith((prefix+verb).lower()):
                            if ((verb.lower().startswith('open')) and linestrs[1].lower().find("with") != -1):
                                print(linestrs[1])
                                intent = "file_share"
                            else:
                                #print(linestrs[1])
                                intent = "file_open"
                            match = True
                            break

            if not match:
                for prefix in prefixWtihVerbWithMy:
                    for verb in searchverbs:
                        if linestrs[1].lower().startswith((prefix+verb).lower()):
                            if ((verb.lower().startswith('show') or verb.lower().startswith('shell')) and linestrs[1].lower().find("with") != -1):
                                print(linestrs[1])
                                intent = "file_share"
                            else:
                                #print(linestrs[1])
                                intent = "file_search"
                            match = True
                            break
            if not match:
                for prefix in prefixWtihVerbWithMy:
                    for verb in downloadverbs:
                        if linestrs[1].lower().startswith((prefix+verb).lower()):
                            intent = "file_download"
                            match = True
                            break
            if not match:
                for prefix in prefixWtihVerbWithMy:
                    for verb in shareverbs:
                        if linestrs[1].lower().startswith((prefix+verb).lower()):
                            intent = "file_share"
                            match = True
                            break

            if not match:
                for prefix in prefixWtihVerbWithMy:
                    for verb in naviverbs:
                        if linestrs[1].lower().find((prefix+verb).lower()) != -1:
                            intent = "file_navigate"
                            match = True
                            break
            if not match:
                for prefix in prefixWtihVerbWithMy:
                    for verb in openverbs:
                        if linestrs[1].lower().find((prefix+verb).lower()) != -1:
                            if ((verb.lower().startswith('open')) and linestrs[1].lower().find("with") != -1):
                                print(linestrs[1])
                                intent = "file_share"
                            else:
                                #print(linestrs[2])
                                intent = "file_open"
                            match = True
                            break

            if not match:
                for prefix in prefixWtihVerbWithMy:
                    for verb in searchverbs:
                        if linestrs[1].lower().find((prefix+verb).lower()) != -1:
                            if ((verb.lower().startswith('show') or verb.lower().startswith('shell')) and linestrs[1].lower().find("with") != -1):
                                print(linestrs[1])
                                intent = "file_share"
                            else:
                                #print(linestrs[2])
                                intent = "file_search"
                            match = True
                            break

            if not match:
                for prefix in prefixWtihVerbWithMy:
                    for verb in downloadverbs:
                        if linestrs[1].lower().find((prefix+verb).lower()) != -1:
                            intent = "file_download"
                            match = True
                            break
            if not match:
                for prefix in prefixWtihVerbWithMy:
                    for verb in shareverbs:
                        if linestrs[1].lower().find((prefix+verb).lower()) != -1:
                            intent = "file_share"
                            match = True
                            break
            '''
            # each one should be exclusive
            intent = 'file_search'
            for prefix in prefixWtihVerbWithMy:
                for verb in naviverbs:
                    #if linestrs[1].lower().startswith((prefix+verb).lower()):
                    if linestrs[1].lower().startswith((prefix+verb).lower()) or  linestrs[1].lower().find((prefix+verb).lower()) != -1:
                        intent = "file_navigate"
                        match = True
                        break
            for prefix in prefixWtihVerbWithMy:
                for verb in openverbs:
                    #if linestrs[1].lower().startswith((prefix+verb).lower()):
                    if linestrs[1].lower().startswith((prefix+verb).lower()) or  linestrs[1].lower().find((prefix+verb).lower()) != -1:
                        intent = "file_open"
                        match = True
                        break

            #for verb in searchverbs:
            #    if linestrs[1].lower().startswith(prefix+verb):
            #       linestrs[2] = "file_search"
            #       match = True
            #       break

            for prefix in prefixWtihVerbWithMy:
                for verb in downloadverbs:
                    #if linestrs[1].lower().startswith((prefix+verb).lower()):
                    if linestrs[1].lower().startswith((prefix+verb).lower()) or  linestrs[1].lower().find((prefix+verb).lower()) != -1:
                        intent = "file_download"
                        match = True
                        break

            for prefix in prefixWtihVerbWithMy:
                for verb in shareverbs:
                    #if linestrs[1].lower().startswith((prefix+verb).lower()):
                    if linestrs[1].lower().startswith((prefix+verb).lower()) or  linestrs[1].lower().find((prefix+verb).lower()) != -1:
                        intent = "file_share"
                        match = True
                        break
            '''

            # default file_search since no file_other in mystuff data
            '''
            if not match  and intent != 'file_other':

                #my stuff all default file_search so no output 
                if intent != linestrs[2]:
                    print(intent)
                    print(linestrs[1])
                linestrs[2] = intent
            '''

            if intent != linestrs[2]:
                updateQuery.add(linestrs[1]+"\t"+linestrs[2] + "\t" + intent)
                #print(linestrs[1])
                #print("orig:"+linestrs[2])
                #print("new:"+intent)
            linestrs[2] = intent
        
        #for i in range(0,repated_time):
        #    Output.append(linestrs[0]+"\t\t"+linestrs[1]+"\t"+linestrs[2]);

        # add
        # "PreviousTurnDomain"
        # "PreviousTurnIntent"
        # as
        # 'TurnNumber', PREVIOUSTURNINTENT, 'query', 'intent',PREVIOUSTURNDOMAIN])
        # append empty at first
        for i in range(0,repated_time):
            Output.append(linestrs[0]+"\t\t"+linestrs[1]+"\t"+linestrs[2]+"\t");

# my stuff is too many so shuffle
print('shuffling');
random.seed(0.1);
random.shuffle(Output);


# to match with teams intent total data
index = 0
with codecs.open('files_mystuff_after_filtering_intent.tsv', 'w', 'utf-8') as fout:
    for item in Output:
        if index < upper_cnt:
            fout.write(item + '\r\n');
            index = index+1
        else:
            break;

with codecs.open('files_mystuff_update_after_filtering_intent.tsv', 'w', 'utf-8') as fout:
    for item in updateQuery:
        fout.write(item + '\r\n');


with codecs.open('files_mystuff_after_filtering_intent_all.tsv', 'w', 'utf-8') as fout:
    for item in Output:
        fout.write(item + '\r\n');

        

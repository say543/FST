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


prefixWtihVerbWithMy =set([
    #empty is also fine
    "",
    "mystuff ",
    "Can you ",
    "can you ",
    "Please ",
    "please ",
    "Can you please ",
    "can you please ",
    "hey cortana please ",
    "Hey cortana please ",
    "cortana please ",
    "cortana ",
    "Hey cortana ",
    "hey cortana ",
    "Hey cortana can you ",
    "hey cortana can you ",
    "i want to ",
    "I want to ",
    "i need to ",
    "I need to ",
    "i have to ",
    "I have to ",
    "i want you to ",
    "I want you to ",
    "i can't ",
    "I can't ",
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
    "hi what's up ",
    "Hi what's up ",
    "hello can you ",
    "Hello can you ",
    "can you help me ",
    "Can you help me ",
    "can you help ",
    "Can you help ",
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
                        "gogo to ",
                        "navigate to ",
                        ])

            openverbs = set(["open ",
                        "opened ",
                        "view ",
                        ])

            '''
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
            '''


            downloadverbs = set(["download ",
                        "downloaded ",
                        "save ",
                        "saved ",
                        ])

            shareverbs = set(["share ",
                        "shared ",
                        ])

            #print(linestrs[1])

            match = False

            # each one should be exclusive
            
            for prefix in prefixWtihVerbWithMy:
                for verb in naviverbs:
                    if linestrs[1].lower().startswith(prefix+verb):
                        linestrs[2] = "file_navigate"
                        match = True
                        break
            for prefix in prefixWtihVerbWithMy:
                for verb in openverbs:
                    if linestrs[1].lower().startswith(prefix+verb):
                        linestrs[2] = "file_open"
                        match = True
                        break

            #for verb in searchverbs:
            #    if linestrs[1].lower().startswith(prefix+verb):
            #       linestrs[2] = "file_search"
            #       match = True
            #       break

            for prefix in prefixWtihVerbWithMy:
                for verb in downloadverbs:
                    if linestrs[1].lower().startswith(prefix+verb):
                        linestrs[2] = "file_download"
                        match = True
                        break

            for prefix in prefixWtihVerbWithMy:
                for verb in shareverbs:
                    if linestrs[1].lower().startswith(prefix+verb):
                        linestrs[2] = "file_share"
                        match = True
                        break

            # default file_search since no file_other in mystuff data
            if not match  and linestrs[2] != 'file_other':

                #my stuff all default file_search so no output 
                #print(linestrs[2])
                #if linestrs[2] != 'file_search':
                #    print(linestrs[1])
                #    print(linestrs[2])
                linestrs[2] = "file_search"
        
        for i in range(0,repated_time):
            Output.append(linestrs[0]+"\t\t"+linestrs[1]+"\t"+linestrs[2]);

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

with codecs.open('files_mystuff_after_filtering_intent_all.tsv', 'w', 'utf-8') as fout:
    for item in Output:
        fout.write(item + '\r\n');

        

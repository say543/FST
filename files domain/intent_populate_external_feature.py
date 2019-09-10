import glob;
import codecs;
import random;
import os


files = ["email_merge_with_crowdsource_intent_only_email.tsv"]
markFile = "email_merge_intent_mark.tsv"
outputs = [];




mark = set()
cnt = 0
with codecs.open(markFile, 'r', 'utf-8') as fin:
        for line in fin:

            # skip head
            if cnt == 0:
                cnt = cnt+1
                continue
            
            line = line.strip();
            if not line:
                continue;

            linestrs = line.split("\t");
            # query \ ExternalFeatureStr

            externalFeature = linestrs[8].replace(";","")
            externalFeature = ''.join(sorted(externalFeature))
            key = linestrs[0]+"\t"+ externalFeature
            #print(key)
            mark.add(key)
            

for file in files:

    outputFile = file.replace('.tsv', '-populate.tsv')
    

   # skip endswith
    if file.endswith("populate.tsv"):
        continue;

    print("replacing: " + file);
    cnt = 0
    with codecs.open(file, 'r', 'utf-8') as fin:
        for line in fin:

            # skip head
            if cnt == 0:
                cnt = cnt+1
                outputs.append(line);
                continue

            orignalline = line
            #print(orignalline)

            # line for preprcessing only
            line = line.strip();
            if not line:
                continue;


            
            linestrs = line.split("\t");


            #print(len(linestrs))

            #print(len(linestrs))

            # no externmal feature to be replaced
            if (len(linestrs)) < 11:
                outputs.append(orignalline);
                continue
            
            externalFeature = linestrs[10].replace(";","")
            externalFeature = ''.join(sorted(externalFeature))
            key = linestrs[1]+"\t"+ externalFeature
            #print(key)

            if key in mark:
                print(line)
                # only keep email as previousTurnDomainInfo
                orignalline = orignalline.replace(linestrs[10],"PreviousTurnDomain(email)")

            #print("add")
            outputs.append(orignalline);



with codecs.open(outputFile, 'w', 'utf-8') as fout:
    for item in outputs:
        fout.write(item);


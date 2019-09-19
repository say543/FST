import glob;
import codecs;
import random;
import os


#outputFile = 'files_slot_training.tsv'
files = glob.glob("*.tsv");
outputs = [];
#outputsWithSource = [];

fileTypeTag={

    # space is important
    # order is important

    ' pptx' : '<file_type> pptx </file_type> ',
    ' ppts' : '<file_type> ppts </file_type> ',
    ' ppt' : '<file_type> ppt </file_type> ',
    ' deck' : '<file_type> deck </file_type> ',
    ' decks' : '<file_type> decks </file_type> ',
    ' presentation' : '<file_type> presentation </file_type> ',
    ' presentations' : '<file_type> presentations </file_type> ',
    ' powerpoint' : '<file_type> powerpoint </file_type> ',
    ' PowerPoint' : '<file_type> PowerPoint </file_type> ',
    ' powerpoints' : '<file_type> powerpoints </file_type> ',
    # add for seperate
    ' power point' : '<file_type> power point </file_type> ',
    
    ' slide' : '<file_type> slides </file_type> ',
    ' slides' : '<file_type> slides </file_type> ',
    ' doc' : '<file_type> doc </file_type> ',
    ' docx' : '<file_type> docx </file_type> ',
    ' docs' : '<file_type> docs </file_type> ',
    # add for upper case
    ' Doc' : '<file_type> Doc </file_type> ',
    ' Docx' : '<file_type> Docx </file_type> ',
    ' Docs' : '<file_type> Docs </file_type> ',

    ' spec' : '<file_type> spec </file_type> ',
    ' excel' : '<file_type> excel </file_type> ',
    ' excels' : '<file_type> excels </file_type> ',
    ' xls' : '<file_type> xls </file_type> ',
    ' xlsx' : '<file_type> xlsx </file_type> ',
    ' spreadsheet' : '<file_type> spreadsheet </file_type> ',
    ' spreadsheets' : '<file_type> spreadsheets </file_type> ',
    ' workbook' : '<file_type> workbook </file_type> ',
    ' worksheet' : '<file_type> worksheet </file_type> ',
    ' csv' : '<file_type> csv </file_type> ',
    ' tsv' : '<file_type> tsv </file_type> ',
    ' note' : '<file_type> note </file_type> ',
    ' notes' : '<file_type> notes </file_type> ',
    ' onenote' : '<file_type> onenote </file_type> ',
    ' onenotes' : '<file_type> onenotes </file_type> ',
    # add for upper case
    ' OneNote' : '<file_type> OneNote </file_type> ',
    ' notebook' : '<file_type> notebook </file_type> ',
    ' notebooks' : '<file_type> notebooks </file_type> ',
    ' pdf' : '<file_type> pdf </file_type> ',
    ' pdfs' : '<file_type> pdfs </file_type> ',
    # add for upper case
    ' PDF' : '<file_type> PDF </file_type> ',
    ' jpg' : '<file_type> jpg </file_type> ',
    ' jpeg' : '<file_type> jpeg </file_type> ',
    ' gif' : '<file_type> gif </file_type> ',
    ' png' : '<file_type> png </file_type> ',
    ' image' : '<file_type> image </file_type> ',
    ' msg' : '<file_type> msg </file_type> ',
    ' ics' : '<file_type> ics </file_type> ',
    ' vcs' : '<file_type> vcs </file_type> ',
    ' vsdx' : '<file_type> vsdx </file_type> ',
    ' vssx' : '<file_type> vssx </file_type> ',
    ' vstx' : '<file_type> vstx </file_type> ',
    ' vsdm' : '<file_type> vsdm </file_type> ',
    ' vssm' : '<file_type> vssm </file_type> ',
    ' vstm' : '<file_type> vstm </file_type> ',
    ' vsd' : '<file_type> vsd </file_type> ',
    ' vdw' : '<file_type> vdw </file_type> ',
    ' vss' : '<file_type> vss </file_type> ',
    ' vst' : '<file_type> vst </file_type> ',
    ' mpp' : '<file_type> mpp </file_type> ',
    ' mpt' : '<file_type> mpt </file_type> ',
    # no mention in spec
    # move it to not tag
    ' word' : '<file_type> word </file_type> ',


    # keep it as tag
    ' picture' : '<file_type> picture </file_type> ',
    ' music' : '<file_type> music </file_type> ',
    ' txt' : '<file_type> txt </file_type> ',
}


for file in files:
    outputFile = file.replace('.tsv', '-fill.tsv')


   # skip endswith
    if file.endswith("fill.tsv"):
        continue;

    print("Filling: " + file);
    with codecs.open(file, 'r', 'utf-8') as fin:

        for line in fin:
            line = line.strip();
            if not line:
                continue;
            
            linestrs = line.split("\t");


            # one line con only match one time
            match = False

            #add space at first for replacement
            slot = " " + linestrs[1] + " "
            for key in reversed(sorted(fileTypeTag.keys())):
                if not match and slot.find(key) != -1:

                    # skip these case because of doc and docx
                    if slot.find("document") != -1 or slot.find("documents") != -1:
                        continue
                    
                    #print(key)
                    match = True
                    slot = slot.replace(key, fileTypeTag[key])

            # after all slot replacement
            # remove head and end spaces 
            slot = slot.strip()


            # output id	query	intent	domain	QueryXml  
            outputs.append("0\t"+linestrs[1]+"\t"+"file_search"+"\t"+"FILES"+"\t"+slot);
            #outputsWithSource.append(line+'\t'+ file);

# no shuffle
#print('shuffling');
#random.seed(0.1);
#random.shuffle(outputs);

#outputs = ['\t'.join(['id', 'query', 'intent', 'domain', 'QueryXml', 'id', '0'])] + outputs;
outputs = ['\t'.join(['id', 'query', 'intent', 'domain', 'QueryXml'])] + outputs;
#outputsWithSource = ['\t'.join(['id', 'query', 'intent', 'domain', 'QueryXml', 'id', '0', 'source'])] + outputsWithSource;




with codecs.open(outputFile, 'w', 'utf-8') as fout:
    for item in outputs:
        fout.write(item + '\r\n');



12/10/2020

//command to output UserFileNamesSpan as feature1
PS E:\qasmlgtools\app\MLGTools> NMLGTools.exe crffeaturize -i query.txt -p joint.featurizer.pipeline.txt -f crf_input_query,UserFileNamesSpan  -s "0" -e "4"


//command to output UserFileNamesSpan as feature1
//command to match a static lexicon as feature 2
PS E:\qasmlgtools\app\MLGTools> NMLGTools.exe crffeaturize -i query.txt -p joint.featurizer.pipeline.txt -f crf_input_query,UserFileNamesSpan,slot.mlg.filekeywordOrfilename.matches  -s "0" -e "4"


12/10/2020

//command to output UserFileNamesSpan as feature1
PS E:\qasmlgtools\app\MLGTools> NMLGTools.exe crffeaturize -i query.txt -p joint.featurizer.pipeline.txt -f crf_input_query,UserFileNamesSpan  -s "0" -e "4"


//command to output UserFileNamesSpan as feature1
//command to match a static lexicon as feature 2
PS E:\qasmlgtools\app\MLGTools> NMLGTools.exe crffeaturize -i query.txt -p joint.featurizer.pipeline.txt -f crf_input_query,UserFileNamesSpan,slot.mlg.filekeywordOrfilename.matches  -s "0" -e "4"

# Hey Cortana Open file called nippy protocol introduction .
[CLS]	
open	
file	(2,0,1.00000000) 
called	
nippy	(1,0,1.00000000) 
protocol	(1,0,1.00000000) 
introduction	(1,0,1.00000000) 

# Hey Cortana Open attachment called nippy protocol introduction .
[CLS]	
open	
attachment	(2,0,1.00000000) 
called	
nippy	(1,0,1.00000000) 
protocol	(1,0,1.00000000) 
introduction	(1,0,1.00000000) 

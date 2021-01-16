from glob import glob
import pandas as pd
import re
from pprint import pprint
import string
#from nltk.tokenize import word_tokenize
from collections import defaultdict
import json
import os
import csv



#target = pd.read_csv('target.tsv', sep='\t', encoding="utf-8")
filebasename = 'files_domain_training_answer'
target = pd.read_csv('../azureml_data/'+filebasename+'.tsv', sep='\t', encoding="utf-8", dtype={
    'domain': object, 'TurnNumber': object, 'PreviousTurnIntent': object, 'query': object})

#update_df = pd.read_csv('../azureml_data/files_domain_training_answer.tsv', sep='\t', encoding="utf-8", dtype={
#    'domain': object, 'TurnNumber': object, 'PreviousTurnIntent': object, 'query': object})



# https://www.kite.com/python/answers/how-to-update-a-value-in-each-row-of-a-pandas-dataframe-in-python#:~:text=Update%20elements%20of%20a%20column,the%20column%20to%20be%20changed.
for i in target.index:
    if target.at[i, "domain"] == 'files':
        target.at[i, "domain"] = 1
    else:
        target.at[i, "domain"] = 0

target.to_csv('../azureml_data/'+filebasename+'_updated.tsv', header=None, index=None, sep='\t')





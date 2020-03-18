# Script to help renaming a bunch of QAS files as well as their mention inside pipelines and configs
# Example Usage:
# python qas_file_renamer.py -s scenario\myscenario\ -t outputdir -r "Microsoft_Threshold_Shell_2:threshold,MV6:version"

import sys
import shutil
import os

from optparse import OptionParser

def istextfile(path):
    text_file_extensions = ['.txt', '.ini', '.tsv']
    return any(path.endswith(extension) for extension in text_file_extensions)

def create_replacement_dict(expression):
    result = {}
    for subexp in expression.split(','):
        key, value = subexp.split(':')
        result[key] = value
    return result
    
def replace(text, replacementdict):
    for (k,v)  in replacementdict.items():
        text = text.replace(k, v)
    return text


def qasFileRenamer(args):
    parser = OptionParser()
    parser.add_option("-s", "--source", dest="sourcedir", help="source directory where all the files are", metavar="FILE")
    parser.add_option("-t", "--target", dest="targetdir", help="target directory where the output will be generated", metavar="FILE")
    parser.add_option("-r", "--replacements", dest="replacements_expression", help="things to replace in file names and inside text files of the form: source:replacement,source2:replacement2", metavar="STRING")
    (options, args) = parser.parse_args(args)
    replacementdict = create_replacement_dict(options.replacements_expression)
    
    try:
        os.makedirs(options.targetdir)
    except:
        sys.stderr.write('%s already exists, not creating.\n' % options.targetdir)
    
    # copy all files from the source dir to target dir with the new names 
    for filename in os.listdir(options.sourcedir):
        sourcepath = os.path.join(options.sourcedir, filename)
        targetpath = os.path.join(options.targetdir, replace(filename, replacementdict))
        if os.path.isfile(sourcepath):
            if istextfile(sourcepath):
                # for text file we search/replace for the given replacement inside the file as well
                with open(sourcepath) as infile, open(targetpath, 'w') as outfile:
                    for line in infile:
                        line = replace(line, replacementdict)
                        outfile.write(line)
            else:
                #binary file we just copy to the new name
                shutil.copyfile(sourcepath, targetpath)

if __name__ == '__main__':
    cmd = ("-s . -t replaced -r cortana_fallback_enus_mv1:cortana_fallback_uwp_enus_mv1").split()
    qasFileRenamer(cmd)

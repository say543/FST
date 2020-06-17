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

    print(options)
    print(args)
    
    replacementdict = create_replacement_dict(options.replacements_expression)
    

    '''
    try:
        os.makedirs(options.targetdir)
    except:
        sys.stderr.write('%s already exists, so override.\n' % options.targetdir)
    '''
    if os.path.exists(options.targetdir):
        print('delete')
        shutil.rmtree(options.targetdir)
    try:
        os.makedirs(options.targetdir)
    except:
        sys.stderr.write('%s already exists, so override.\n' % options.targetdir)
    
    # copy all files from the source dir to target dir with the new names 
    for filename in os.listdir(options.sourcedir):

        print(filename)
        
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
    #cmd = ("-s . -t replaced -r cortana_fallback_enus_mv1:cortana_fallback_uwp_enus_mv1").split()

    #UWP SVM
    #cmd = ("-s ..\\output_domain_03232020v1\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()

    #answer domain svm
    #cmd = ("-s ..\\output_domain_answer_03262020v1\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()
    #cmd = ("-s ..\\output_domain_answer_03272020v1\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()
    #cmd = ("-s ..\\output_domain_answer_03272020v2\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()
    #cmd = ("-s ..\\output_domain_answer_03272020v3\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()
    #cmd = ("-s ..\\output_domain_answer_03272020v4\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()

    #cmd = ("-s ..\\output_domain_answer_04092020v1\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()
    #cmd = ("-s ..\\output_domain_answer_04102020v1_using_pcfg_v2\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()
    #cmd = ("-s ..\\output_domain_answer_04102020V2_use_pcfg_v2_domain\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()

    #cmd = ("-s ..\\output_domain_answer_04132020v1_use_pcfg_v2_domain\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()
    #cmd = ("-s ..\\output_domain_answer_04132020v1_use_pcfg_v2_domain\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv3").split()

    #cmd = ("-s ..\\output_domain_answer_04142020v1_use_pcfg_v2_domain\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()
    #cmd = ("-s ..\\output_domain_answer_04162020v1_use_pcfg_v3_domain\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv3").split()
    #cmd = ("-s ..\\output_domain_answer_04162020v2_use_pcfg_v4_domain\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv3").split()

    #cmd = ("-s ..\\output_domain_answer_04172020v1_use_pcfg_v3_domain\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv3").split()
    #cmd = ("-s ..\\output_domain_answer_04172020v2_use_pcfg_v3_domain\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()


    #cmd = ("-s ..\\output_domain_answer_04172020v3_use_pcfg_v3_domain_cut_some_uesless_domains\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()
    #cmd = ("-s ..\\output_domain_answer_04202020v1_use_pcfg_v3_domain_cut_some_uesless_domains\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv3").split()

    #cmd = ("-s ..\\output_domain_answer_04212020v1_use_pcfg_v3_domain_cut_some_uesless_domains_negative_cases\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()

    #cmd = ("-s ..\\output_domain_answer_04212020v2_use_pcfg_v3_domain_cut_some_uesless_domains_negative_cases\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv3").split()

    # this one checked in to all
    #cmd = ("-s ..\\output_domain_answer_04242020v1_use_pcfg_v3_domain_cut_some_uesless_domains_negative_cases\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv3").split()


    #cmd = ("-s ..\\output_domain_answer_05012020v1_use_pcfg_v3_domain_cut_some_uesless_domains_negative_cases\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()

    #cmd = ("-s ..\\output_domain_answer_05012020v2_use_pcfg_v3_domain_cut_some_uesless_domains_negative_cases\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()
    #cmd = ("-s ..\\output_domain_answer_05012020v3_use_pcfg_v3_domain_cut_some_uesless_domains_negative_cases\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()


    #cmd = ("-s ..\\output_adhoc_slot_05052020v5\\slot_lccrf\\final -t replaced -r files_enus_mv1:files_enus_mv3").split()

    ###########
    ### checked in domian model ###
    ###########
    #cmd = ("-s ..\\output_domain_answer_05012020v3_use_pcfg_v3_domain_cut_some_uesless_domains_negative_cases\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()
    #cmd = ("-s ..\\output_domain_answer_05012020v3_use_pcfg_v3_domain_cut_some_uesless_domains_negative_cases\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv3").split()

    #cmd = ("-s ..\\output_domain_answer_05142020v2_use_pcfg_v3_domain_cut_some_uesless_domains_negative_cases\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv3").split()


    #cmd = ("-s ..\\output_domain_answer_05242020v1_300\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv3").split()

    ###########
    ### checked in intent model ###
    ###########

    #cmd = ("-s ..\\output_adhoc_intent_05042020v3\\intent_svm\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()
    #cmd = ("-s ..\\output_adhoc_intent_05042020v3\\intent_svm\\final -t replaced -r files_enus_mv1:files_enus_mv3").split()


    #cmd = ("-s ..\\output_adhoc_intent_05292020v2\\intent_svm\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()

    cmd = ("-s ..\\output_adhoc_intent_06112020v4\\intent_svm\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()


    ###########
    ### checked in slot model ###
    ###########

    # this version use some uwp random set query(but their annotation might be wrong, update it in the future)
    #cmd = ("-s ..\\output_adhoc_slot_05062020v3\\slot_lccrf\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()


    #cmd = ("-s ..\\output_adhoc_slot_05202020v2\\slot_lccrf\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()


    #cmd = ("-s ..\\output_adhoc_slot_05272020v1\\slot_lccrf\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()

    #cmd = ("-s ..\\output_adhoc_slot_05272020v1\\slot_lccrf\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()

    #cmd = ("-s ..\\output_adhoc_slot_05282020v1\\slot_lccrf\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()


    #cmd = ("-s ..\\output_adhoc_slot_05292020v1\\slot_lccrf\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()


    # this one attached / attachment might have wrong scheme, need to revisit in the future
    # golden queire might need to check
    #cmd = ("-s ..\\output_adhoc_slot_06052020v1\\slot_lccrf\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()
    


    ###########
    ### other try fine-tune experience
    ###########

    #cmd = ("-s ..\\output_adhoc_intent_06092020v1\\intent_svm\\final -t replaced -r files_enus_mv1:files_enus_mv3").split()

    #cmd = ("-s ..\\output_adhoc_slot_06052020v1\\slot_lccrf\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()


    # basically add all
    #cmd = ("-s ..\\output_domain_answer_05182020v1_use_pcfg_v3_domain_cut_some_uesless_domains_negative_cases\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv3").split()

    #cmd = ("-s ..\\output_domain_answer_05202020v2_use_pcfg_v3_domain_cut_some_uesless_domains_negative_cases\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv3").split()


    #cmd = ("-s ..\\output_domain_answer_05232020v1_100\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv3").split()

    #cmd = ("-s ..\\output_domain_answer_05232020v1_200\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv3").split()

    #cmd = ("-s ..\\output_domain_answer_05232020v1_300\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv3").split()



    #cmd = ("-s ..\\output_domain_answer_05242020v1_100\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv3").split()


    #cmd = ("-s ..\\output_domain_answer_05242020v1_200\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv3").split()

    

    #cmd = ("-s ..\\output_domain_answer_05222020v1_1000\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv3").split()

    #cmd = ("-s ..\\output_domain_answer_05222020v1_1500\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv3").split()

    #cmd = ("-s ..\\output_domain_answer_05222020v1_2000\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv3").split()

    
    #cmd = ("-s ..\\output_domain_answer_05222020v1_2500\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv3").split()

    #cmd = ("-s ..\\output_domain_answer_05222020v1_1500\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv3").split()


    #cmd = ("-s ..\\output_adhoc_slot_05212020v1\\slot_lccrf\\final -t replaced -r files_enus_mv1:files_enus_mv3").split()


    #cmd = ("-s ..\\output_adhoc_slot_05202020v2\\slot_lccrf\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()

    #cmd = ("-s ..\\output_adhoc_slot\\canonical_processor\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()


    #cmd = ("-s E:\\files_mv1_mv3_ready_for_teams_flight -t replaced -r files_enus_mv1:files_enus_mv3").split()

    # share intent svm
    #cmd = ("-s ..\\output_adhoc_intent\\intent_svm\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()

    # one work example
    #cmd = ("-s ..\\output_adhoc_slot\\slot_lccrf\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()


    #rename files from batstatics to v3 for new domain to train
    #cmd = ("-s ..\\..\\..\\files_bat_model_future_b2 -t replaced -r files_enus_mv1:files_enus_mv3").split()
    
    qasFileRenamer(cmd)

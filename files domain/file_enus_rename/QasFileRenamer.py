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

        # ignore those filename
        if filename == 'filekeys.txt' or filename == 'filekeys_chiecha.txt' or  filename == 'filekeys_chiecha_07292020v1.txt' or filename == 'files_domain_training_contexual_answer.tsv':
            continue

        # ignore those filename
        if filename == 'filekeys.txt' or filename == 'filekeys_intent_chiecha.txt' or filename == 'files_intent_training_contexual_answer.tsv':
            continue
        
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

    #cmd = ("-s ..\\output_domain_answer_07132020v1_300\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()


    #cmd = ("-s ..\\output_domain_answer_07162020v1_300\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv3").split()


    
    #cmd = ("-s ..\\output_domain_answer_10202020v2_300\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()

    ###########
    ### checked in intent model ###
    ###########

    #cmd = ("-s ..\\output_adhoc_intent_05042020v3\\intent_svm\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()
    #cmd = ("-s ..\\output_adhoc_intent_05042020v3\\intent_svm\\final -t replaced -r files_enus_mv1:files_enus_mv3").split()


    #cmd = ("-s ..\\output_adhoc_intent_05292020v2\\intent_svm\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()

    #cmd = ("-s ..\\output_adhoc_intent_06112020v4\\intent_svm\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()

    #cmd = ("-s ..\\output_adhoc_intent_07012020v1\\intent_svm\\final -t replaced -r files_enus_mv1:files_enus_mv3").split()

    #cmd = ("-s ..\\output_adhoc_intent_07082020v1\\intent_svm\\final -t replaced -r files_enus_mv1:files_enus_mv3").split()

    #cmd = ("-s ..\\output_adhoc_intent_08212020v1\\intent_svm\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()
    #cmd = ("-s ..\\output_adhoc_intent_10132020v1\\intent_svm\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()


    ###########
    ### checked in slot model ###
    ###########

    # this version use some uwp random set query(but their annotation might be wrong, update it in the future)
    #cmd = ("-s ..\\output_adhoc_slot_05062020v3\\slot_lccrf\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()

    #cmd = ("-s ..\\output_domain_synthetic_06222020v1\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv3").split()

    #cmd = ("-s ..\\output_domain_synthetic_06222020v2\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()

    #cmd = ("-s ..\\output_domain_synthetic_06222020v3\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv3").split()

    #cmd = ("-s ..\\output_domain_synthetic_06232020v4\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv3").split()

    #cmd = ("-s ..\\output_domain_synthetic_06232020v5\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv3").split()

    # cloud test this
    #cmd = ("-s ..\\output_domain_synthetic_06242020v3\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()

    #cmd = ("-s ..\\output_domain_synthetic_06252020v1\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()


    # kashan replicate models
    #cmd = ("-s E:\\kathan\\FileTest\\exp_06262020v1\\domain_svm\\final -t replaced -r cortana_files_enus_mv1:files_enus_mv1").split()

    
    
    #cmd = ("-s ..\\output_domain_synthetic_06242020v5\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv3").split()

    #cmd = ("-s ..\\output_adhoc_slot_05202020v2\\slot_lccrf\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()


    #cmd = ("-s ..\\output_adhoc_slot_05272020v1\\slot_lccrf\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()

    #cmd = ("-s ..\\output_adhoc_slot_05272020v1\\slot_lccrf\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()

    #cmd = ("-s ..\\output_adhoc_slot_05282020v1\\slot_lccrf\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()


    #cmd = ("-s ..\\output_adhoc_slot_05292020v1\\slot_lccrf\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()


    # this one attached / attachment might have wrong scheme, need to revisit in the future
    # golden queire might need to check
    #cmd = ("-s ..\\output_adhoc_slot_06052020v1\\slot_lccrf\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()


    #cmd = ("-s ..\\output_adhoc_slot_06182020v1\\slot_lccrf\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()


    #cmd = ("-s ..\\output_adhoc_slot_08192020v1\\slot_lccrf\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()

    #cmd = ("-s ..\\output_adhoc_slot_08202020v1\\slot_lccrf\\final -t replaced -r files_enus_mv1:files_en4us_mv2").split()

    #cmd = ("-s ..\\output_adhoc_slot_08202020v2\\slot_lccrf\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()

    #cmd = ("-s ..\\output_adhoc_slot_09142020v1\\slot_lccrf\\final -t replaced -r files_enus_mv1:files_enus_mv5").split()

    #cmd = ("-s ..\\output_adhoc_slot_09142020v2\\slot_lccrf\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()
    #cmd = ("-s ..\\output_adhoc_slot_09242020v2\\slot_lccrf\\final -t replaced -r files_enus_mv1:files_enus_mv3").split()

    #cmd = ("-s ..\\output_adhoc_slot_10012020v4\\slot_lccrf\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()


    ###########
    ### other try fine-tune experience
    ###########

    #cmd = ("-s ..\\..\\..\\files_enus_mv5 -t replaced -r files_enus_mv3:files_enus_mv5").split()


    #cmd = ("-s ..\\output_domain_answer_pergra_07212020v1\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()
    #cmd = ("-s ..\\output_domain_answer_pergra_07242020v2\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv3").split()
    #cmd = ("-s ..\\output_domain_answer_pergra\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()

    # aether for domain
    #cmd = ("-s ..\\output_domain_answer_pergra_07242020v3 -t replaced -r files_enus_mv1:files_enus_mv2").split()
    #cmd = ("-s ..\\output_domain_answer_pergra_07292020v1 -t replaced -r files_enus_mv1:files_enus_mv2").split()
    #cmd = ("-s ..\\output_domain_answer_pergra_08032020v1 -t replaced -r files_enus_mv1:files_enus_mv2").split()
    #cmd = ("-s ..\\output_domain_answer_pergra_08032020v2 -t replaced -r files_enus_mv1:files_enus_mv2").split()
    #cmd = ("-s ..\\output_domain_answer_pergra_08032020v3 -t replaced -r files_enus_mv1:files_enus_mv3").split()
    #cmd = ("-s ..\\output_domain_answer_pergra_08042020v1 -t replaced -r files_enus_mv1:files_enus_mv2").split()

    #cmd = ("-s ..\\output_domain_answer_pergra_08042020v2 -t replaced -r files_enus_mv1:files_enus_mv2").split()
    #cmd = ("-s ..\\output_domain_answer_pergra_08042020v3 -t replaced -r files_enus_mv1:files_enus_mv3").split()
    #cmd = ("-s ..\\output_domain_answer_pergra_08042020v4 -t replaced -r files_enus_mv1:files_enus_mv2").split()

    #cmd = ("-s ..\\output_domain_answer_pergra_08052020v1 -t replaced -r files_enus_mv1:files_enus_mv2").split()
    #cmd = ("-s ..\\output_domain_answer_pergra_08062020v1 -t replaced -r files_enus_mv1:files_enus_mv3").split()

    #cmd = ("-s ..\\output_domain_answer_pergra_08062020v2 -t replaced -r files_enus_mv1:files_enus_mv2").split()

    #cmd = ("-s ..\\output_domain_answer_pergra_08102020v1 -t replaced -r files_enus_mv1:files_enus_mv3").split()
    #cmd = ("-s ..\\output_domain_answer_pergra_08112020v1 -t replaced -r files_enus_mv1:files_enus_mv2").split()

    #cmd = ("-s ..\\output_domain_answer_pergra_08122020v1 -t replaced -r files_enus_mv1:files_enus_mv3").split()

    #cmd = ("-s ..\\output_domain_answer_pergra_08122020v2 -t replaced -r files_enus_mv1:files_enus_mv2").split()

    #cmd = ("-s ..\\output_domain_answer_pergra_08152020v1 -t replaced -r files_enus_mv1:files_enus_mv3").split()
    #cmd = ("-s ..\\output_domain_answer_pergra_08182020v1 -t replaced -r files_enus_mv1:files_enus_mv3").split()

    #cmd = ("-s ..\\output_domain_answer_pergra_08192020v1 -t replaced -r files_enus_mv1:files_enus_mv3").split()

    #cmd = ("-s ..\\output_domain_answer_pergra_08192020v2 -t replaced -r files_enus_mv1:files_enus_mv3").split()
    #cmd = ("-s ..\\output_domain_answer_pergra_08192020v3 -t replaced -r files_enus_mv1:files_enus_mv2").split()


    #cmd = ("-s ..\\output_domain_answer_pergra_08202020v1 -t replaced -r files_enus_mv1:files_enus_mv2").split()

    #cmd = ("-s ..\\output_domain_answer_pergra_08202020v2 -t replaced -r files_enus_mv1:files_enus_mv2").split()

    #cmd = ("-s ..\\output_domain_answer_pergra_08212020v1 -t replaced -r files_enus_mv1:files_enus_mv2").split()

    #cmd = ("-s ..\\output_domain_answer_pergra_08212020v2 -t replaced -r files_enus_mv1:files_enus_mv3").split()

    #cmd = ("-s ..\\output_domain_answer_pergra_08212020v3 -t replaced -r files_enus_mv1:files_enus_mv3").split()

    #cmd = ("-s ..\\output_domain_answer_pergra_08212020v4 -t replaced -r files_enus_mv1:files_enus_mv2").split()


    #cmd = ("-s ..\\output_domain_answer_pergra_08232020v1 -t replaced -r files_enus_mv1:files_enus_mv2").split()

    #cmd = ("-s ..\\output_domain_answer_pergra_08242020v1 -t replaced -r files_enus_mv1:files_enus_mv5").split()

    # local
    # doamin
    #cmd = ("-s ..\\output_domain_answer_pergra_bo_08122020v2\\domain_svm\\final -t replaced -r files_enus_mv1:files_enus_mv3").split()

    # local intent
    #cmd = ("-s ..\\output_intent_answer_pergra\\intent_svm\\final -t replaced -r files_enus_mv1:files_enus_mv3").split()
    #cmd = ("-s ..\\output_intent_answer_pergra_08172020v1\\intent_svm\\final -t replaced -r files_enus_mv1:files_enus_mv3").split()

    #cmd = ("-s ..\\output_intent_answer_pergra_08182020v1\\intent_svm\\final -t replaced -r files_enus_mv1:files_enus_mv3").split()
    #cmd = ("-s ..\\output_intent_answer_pergra_08182020v2\\intent_svm\\final -t replaced -r files_enus_mv1:files_enus_mv3").split()
    #cmd = ("-s ..\\output_intent_answer_pergra_08182020v3\\intent_svm\\final -t replaced -r files_enus_mv1:files_enus_mv3").split()
    



    # aether for intent
    #cmd = ("-s ..\\output_intent_answer_pergra\\intent_svm\\final -t replaced -r files_enus_mv1:files_enus_mv3").split()
    #cmd = ("-s ..\\output_intent_answer_pergra_08182020v3\\ -t replaced -r files_enus_mv1:files_enus_mv3").split()
    #cmd = ("-s ..\\output_intent_answer_pergra_08182020v4\\ -t replaced -r files_enus_mv1:files_enus_mv3").split()
    #cmd = ("-s ..\\output_intent_answer_pergra_08182020v5\\intent_svm\\final -t replaced -r files_enus_mv1:files_enus_mv3").split()

    #cmd = ("-s ..\\output_intent_answer_pergra_08182020v6\\intent_svm\\final -t replaced -r files_enus_mv1:files_enus_mv3").split()


    #cmd = ("-s ..\\output_intent_answer_pergra_08182020v7\\intent_svm\\final -t replaced -r files_enus_mv1:files_enus_mv2").split()

    #cmd = ("-s ..\\output_intent_answer_pergra_08192020v1\\intent_svm\\final -t replaced -r files_enus_mv1:files_enus_mv3").split()

    #cmd = ("-s ..\\output_intent_answer_pergra_08192020v2\\intent_svm\\final -t replaced -r files_enus_mv1:files_enus_mv3").split()


    cmd = ("-s ..\\output_intent_answer_pergra_11062020v1\\intent_svm\\final -t replaced -r files_enus_mv1:files_enus_mv5").split()

    
    #cmd = ("-s ..\\output_adhoc_slot_07092020v1\\slot_lccrf\\final -t replaced -r files_enus_mv1:files_enus_mv3").split()



    #cmd = ("-s E:\\files_bat_model_before_add_intent_model -t replaced -r files_enus_mv1:files_enus_mv3").split()

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

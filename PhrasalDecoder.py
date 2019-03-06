#!/usr/bin/env python

import argparse
import codecs
import os
from os import chdir, listdir
from os.path import abspath, dirname, exists, isfile, join, split

import shutil
import subprocess
import distutils.dir_util

def run():
    for intent in intents:
        input_folder = join(inputdir, intent, 'input')
        output_folder = join(inputdir, intent, 'output')

        run_par_intent(input_folder, output_folder, intent)


def run_par_intent(input_folder, output_folder, intent_name):
    """
    Execute PhrasalDecoder.exe for one intent
    """
    if not exists(output_folder):
        os.makedirs(output_folder)

    print("executing paraphrase on intent: " + intent_name)
    para_output_file_name = intent_name+'.para'

    config_for_user = load_config(intent_name, input_folder, output_folder, para_output_file_name)

    # the 'final' folder in paraphrase folder is required for the script to execute.
    # so need to copy over the updated configuration file
    # 1. copy new config file into this final folder
    # 2. run the script
    # 3. remove the copied configuration file
    shutil.copy(config_for_user, paraphrase_source_folder)
    config = abspath(join(paraphrase_source_folder, split(abspath(config_for_user))[-1]))

    decoder = join(paraphrase_tool_folder, "PhrasalDecoder.exe")
    with open('decoder_stderr.log', 'w') as f_stderr:
        subprocess.run([decoder, config], stderr=f_stderr)
    
    os.remove(config)

    print("Finish [Executing PhrasalDecoder.exe   ]")

    # Cleaning output file format
    output_file = abspath(join(output_folder, intent_name+'.txt'))
    clean_file(abspath(join(output_folder, para_output_file_name)), output_file)
    print("Output: " + abspath(join(output_folder, intent_name+'.txt')))

    return


def load_config(intent_name, input_folder, output_folder, output_file_name, nbestlen=5):
    """
    Build configuration for given intent.
    :param intent_name: the intent name
    :param input_folder: input directory where has the input data
    :param output_file_name: output configuration file name
    :param nbestlen: number of best list will be output
    :return: the absolute path of the output configuration file.
    """
    output_config_file = abspath(join(input_folder, intent_name+'.para.config'))

    custConfig = dict()
    custConfig['InputDirectory'] = abspath(input_folder)
    custConfig['OutputDirectory'] = abspath(output_folder)
    custConfig['Logfile'] = abspath(join(dirname(input_folder), 'paraphrase.log'))
    custConfig['SourceFile'] = intent_name + '.txt'
    custConfig['ReferenceFile'] = intent_name + '.txt'
    custConfig['OutputFile'] = output_file_name
    custConfig['NBestLength'] = str(nbestlen)

    with open(output_config_file, 'w') as outputconf:
        for pair in custConfig.items():
            outputconf.write(' = '.join(pair) + '\n')

        outputconf.write('\n')

        with open(join(paraphrase_source_folder, 'ENZ.ENU.General.decoder.config'), 'r') as configfile:
            for line in configfile:
                if all([keyword not in line for keyword in custConfig.keys()]):
                    outputconf.write(line.strip() + '\n')

    return output_config_file


def clean_file(inputfilepath, outputfilepath):
    """
    Remove meaningless lines and extra spaces
    :param inputfilepath: file that needs to be reformatted
    :param outputfilepath: output file path
    """
    with codecs.open(inputfilepath, 'r', encoding='utf-16') as fh:
        with codecs.open(outputfilepath, 'w', encoding='utf-8') as outputfile:
            for row in fh:
                l = clean_line(row)
                if l:
                    outputfile.write(l + '\n')

    return


def clean_line(line):
    """
    remove extra spaces
    :param line: input string, output from PhrasalDecoder.exe
    :return: reformatted line
    """
    if (not line.strip()) or "@@" in line or isfloat(line):
        line = ''
    else:
        line = line.replace('< ', '<').replace(' >', '>')
        line = line.replace(' .', '.').replace(' ?', '?').replace(' \'', '\'').replace(' !', '!')

    return line


def isfloat(s):
    """
    varify if a string can be converted to float number
    :param s: input string
    :return: if s can be converted to float number
    """
    try:
        float(s)
    except ValueError:
        return False

    return True


def load_intent_list(intent_file_path):
    """
    Load list of the intents that will be kept
    :param filepath: input file path
    :return: intent whitelist
    """

    with open(intent_file_path, 'r') as filehandle:
        loaded_list = filehandle.read().splitlines()

    return loaded_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run PhrasalDecoder script. ')
    parser.add_argument('--inputdir', dest='inputfolder', required=True,
                        help='directory contains input data, for pipeline purpose the folder name should be an intent')
    parser.add_argument('--paraphrasetool', dest='toolfolder', required=True,
                        help='directory of paraphrase')
    parser.add_argument('--paraphrasesource', dest='paraphrasesource', required=True,
                        help='directory of paraphrase')
    parser.add_argument('--nbestlen', dest='nbestlen',
                        help='number of best list to be retrieved, this is used by PhrasalDecoder.exe')
    parser.add_argument('--sources', dest='sources', help='folder that contains all sources of this intent')
    parser.add_argument('--intentwhitelist', dest='intentwhitelist', help='intent list that will be kept for the rest'
                                                                          'of the pipeline.')
    parser.add_argument('--outputdir', dest='outputdir', help='output directory. ')

    args = parser.parse_args()

    inputdir = abspath(args.inputfolder)
    paraphrase_tool_folder = abspath(args.toolfolder)
    paraphrase_source_folder = abspath(args.paraphrasesource)
    intentwhitelist = load_intent_list(args.intentwhitelist) if args.intentwhitelist else None

    if intentwhitelist:
        intents = [f for f in listdir(inputdir) if f in intentwhitelist]
    else:
        intents = [f for f in listdir(inputdir) if f in load_intent_list(join(args.sources, 'intents.txt'))]

    run()
    
    # copy all files from input directory to output directory
    output_dir_path = abspath(args.outputdir)

    distutils.dir_util.copy_tree(inputdir, output_dir_path)

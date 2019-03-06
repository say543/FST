import argparse
import codecs
from os import listdir
from os.path import abspath, isfile, join, split, splitext
import random
import shutil


def expand(input_dir, outputfilepath, count):
    """
    Lexicon expansion.
    :param input_dir: input directory
    :param outputfilepath: output file path
    :return: filepath
    """

    onlyfiles = [abspath(join(input_dir, fn, 'output', fn+'.txt')) for fn in listdir(input_dir)
                 if ((not intentwhitelist) and fn in intent_list) or (intentwhitelist and fn in intentwhitelist)
                 and isfile(join(input_dir, fn, 'output', fn+'.txt'))]

    with codecs.open(outputfilepath, 'w', encoding='utf-8') as outputfile:
        outputfile.write('\t'.join(['ID', 'Query', 'Intent', 'Domain', 'SlotString']) + '\n')
        for f in onlyfiles:
            print("loading: " + f)
            intent = splitext(split(f)[-1])[0]
            with codecs.open(f, 'r') as inputfile:
                for line in inputfile:
                    line = line.strip().replace(' .', '.').replace(' ?', '?').replace(' \'', '\'').replace(' !', '!')
                    for [untagged, tagged] in sentence_expansion(line, count):
                        outputfile.write('\t'.join(['0', untagged, intent, domain, tagged]) + '\n')

    return outputfilepath


def sentence_expansion(rawseentence, cnt):
    """
    Lexicon expansion
    :param rawseentence: raw sentence
    :param cnt: number of expansion for the given sentence
    :return: expanded sentence list
    """

    indexes = [i for i, ltr in enumerate(rawseentence) if ltr == '>' or ltr == '<']
    slotlist = [rawseentence[indexes[i]: indexes[i+1] + 1] for i in range(0, len(indexes), 2)]
    output = []

    for i in range(cnt):
        # reset
        tmp = rawseentence
        for s in slotlist:
            tmp = tmp.replace(s, resemble(s))

        if tmp.strip():
            untaggedsentence = tmp
            tmp_indexes = [i for i, ltr in enumerate(untaggedsentence) if ltr == '>' or ltr == '<']
            tmp_slotlist = [untaggedsentence[tmp_indexes[i]: tmp_indexes[i + 1] + 1] for i in range(0, len(tmp_indexes), 2)]

            for s in tmp_slotlist:
                untaggedsentence = untaggedsentence.replace(s, '')

            untaggedsentence = untaggedsentence.strip().replace(' .', '.').replace(' ?', '?').replace(' \'', '\'').replace(' !', '!')
            output.append([untaggedsentence, tmp])

    return output


def resemble(tag):
    """
    Resemble tagged slot string with randomly picked slot value
    :param tag: tag
    :return: tagged slot string
    """

    slotname = tag[1:len(tag)-1]
    val = random.choice(lexicon_dict[slotname])
    endtag = tag[:1] + '/' + tag[1:]

    return ' '.join([tag, val, endtag])


def load_lexicon_source(source_dir):
    """
    Load lexicon sources into a dictionary
    :param source_dir: source directory
    :return: lexicon dictionary
    """

    onlyfiles = [f for f in listdir(source_dir) if isfile(join(source_dir, f)) and splitext(f)[0] in slot_list]

    lexicon_gallery = {}
    for f in onlyfiles:
        print('[Loading Lexicon: ] ' + f)
        with codecs.open(join(source_dir, f), 'r', encoding='utf-8') as lexicfile:
            lexicon_gallery[splitext(f)[0]] = lexicfile.read().splitlines()

    return lexicon_gallery


def load_source(filepath):
    """
    Load a file into a list
    :param filepath: input source file path
    :return: list from the source file
    """

    with open(filepath, 'r') as filehandle:
        loaded_list = filehandle.read().splitlines()

    return loaded_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lexicon expansion')
    parser.add_argument('--sources', dest='sources', help='')
    parser.add_argument('--inputdir', dest='inputdir', help='')
    parser.add_argument('--count', dest='count', type=int, default=5, help='')
    parser.add_argument('--intentwhitelist', dest='intentwhitelist', help='intent list that will be kept for the rest'
                                                                          'of the pipeline.')
    parser.add_argument('--output', dest='output', help='output file name. ')

    args = parser.parse_args()
    sources_path = args.sources

    intentwhitelist = load_source(args.intentwhitelist)

    domain = load_source(join(sources_path, 'names.txt'))[0].strip()
    intent_list = load_source(join(sources_path, 'intents.txt'))
    slot_list = load_source(join(sources_path, 'slots.txt'))
    lexicon_dict = load_lexicon_source(sources_path)

    output_file_path = join(args.inputdir, 'data.tsv')
    expand(args.inputdir, output_file_path, args.count)

    # copy all files from input directory to output directory
    print("[Copy]" + output_file_path + " -> " + abspath(args.output))
    shutil.copy(output_file_path, abspath(args.output))

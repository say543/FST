#!/usr/bin/env python

import argparse
import csv
import math
import os
from os.path import join, exists, realpath
import random


def split(filename, outputdir, percentile):
    """
    Randomly picking a certain percentage of data and output them into corresponding intent folder.
    :param filename: input file
    :param outputdir: output directory
    :param percentile: the percentage of data that will be picked
    :return: None
    """
    data = DataByIntent()

    with open(filename, 'r', newline='') as inputfile:
        reader = csv.DictReader(inputfile, dialect='excel-tab')
        for row in reader:
            if (not intentwhitelist) or row['Intent'] in intentwhitelist:
                data.add(row)

    picked = randompick(data, percentile)
    if picked:
        picked.write(outputdir)
    elif not exists(outputdir):
        os.makedirs(outputdir)

    return


def randompick(typeddata, percentile):
    """
    Randomly pick a certain percentage of data out of the given data set.
    :param typeddata: data source
    :param percentile: percentage that will be picked
    :return: picked data
    """

    if typeddata is None or typeddata.count() == 0:
        return

    picked = DataByIntent()
    for intent in typeddata.data:
        target = math.floor(typeddata.count(intent)*float(percentile)/100)
        cnt = target if target >= 1 else typeddata.count(intent)

        data = typeddata.getdata(intent)
        random.shuffle(data)
        picked.data[intent] = data[:cnt]

    return picked


class DataByIntent:
    """
        A class represent typed data
        {intent_name: [dataEntry, ... ]}
    """

    def __init__(self):
        self.data = {}

    def add(self, rawdata):
        """
        Add raw data to corresponding intent
        :param rawdata: raw data
        :return: None
        """

        intent = rawdata['Intent']
        values = self.data.get(intent, [])
        values.append(DataEntry(rawdata))

        self.data[intent] = values
        return

    def getdata(self, intent):
        """
        Retrieve data with the given intent,
        if doesn't exist, return empty list
        :param intent: intent name
        :return: data that with the given intent
        """

        return self.data.get(intent, [])

    def count(self, intent=None):
        """
        Amount of data with the given intent,
        if not specified, return the count of all intents
        :param intent: intent name
        :return: data count of the given intent
        """

        if intent is None:
            return sum([len(i) for i in self.data.values()])
        else:
            return len(self.data.get(intent, []))

    def write(self, outputdir):
        """
        Output data into intent separated folders
        :param outputdir: output directory
        :return: None
        """

        for intent, vals in self.data.items():
            intentdir = join(outputdir, intent, 'input')

            if not exists(intentdir):
                os.makedirs(intentdir)

            with open(join(intentdir, intent+'.txt'), 'w', encoding='utf-16') as outfilehandle:
                for v in vals:
                    outfilehandle.write(v.slotnamereplace() + "\n")

        return


class DataEntry:
    """
        Represent one piece data entry.
    """

    def __init__(self, rawData):
        self.domain = rawData["Domain"]
        self.intent = rawData["Intent"]
        self.SlotString = rawData["SlotString"] if "SlotString" in rawData else rawData["SlotStringXML"]

    def slotnamereplace(self):
        """
        Replace tagged slot to only tag.
        e.g. <tagname> tagvalue </tagname> => <tagname>
        :return: replaced tagged sentence
        """

        slotstring = self.SlotString
        indexes = [i for i, ltr in enumerate(self.SlotString) if ltr == '>']
        substrings = [self.SlotString[indexes[i] + 1:indexes[i + 1] + 1] for i in range(0, len(indexes), 2)]
        for ss in substrings:
            slotstring = slotstring.replace(ss, '')

        return slotstring


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
    # filename = "C:/Users/wujie/PycharmProjects/LanguageFactory/data/experiment/Train_2017-07-18.tsv"
    # split(filename, 100)

    parser = argparse.ArgumentParser(description='Randomly picking certain percentage of data, '
                                                 'separate data by intents. ')

    parser.add_argument('--input', dest='inputfile', required=True,
                        help='Input file path, data should be tagged utterances. ')
    parser.add_argument('--output', dest='outputdir',
                        help='Output directory, if not specified, ')
    parser.add_argument('--pick', dest='percentage', type=int, default=100,
                        help='the picking a certain percentage of the data, if not specified all data will be taken. ')
    parser.add_argument('--intentwhitelist', dest='intentwhitelist', help='intent list that will be kept for the rest'
                                                                          'of the pipeline.')
    # parser.add_argument('passthrough', nargs='+', help='for pass through intent whitelist from aether job. ')

    args = parser.parse_args()
    intentwhitelist = load_intent_list(args.intentwhitelist) if args.intentwhitelist else None
    split(args.inputfile, args.outputdir, args.percentage)

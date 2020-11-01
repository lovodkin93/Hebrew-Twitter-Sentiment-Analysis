import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import stanza
import requests
import json
from time import sleep
from pandas.io.json import json_normalize
from enum import Enum
import os
import re


class Processor:
    def __init__(self):
        self.heb_nlp = stanza.Pipeline(lang='he', processors='tokenize,mwt,pos,lemma,depparse')
        # replace MY_TOKEN with the token you got from the langndata website
        self.yap_token = "4def4deab459ce5707bd0d004da4c484"

    def get_analysis(self, text):
        text += " XX"
        doc = self.heb_nlp(text)
        lst = []
        for sen in doc.sentences:
            for token in sen.tokens:
                for word in token.words:
                    features = [(word.text,
                                 word.lemma,
                                 word.upos,
                                 word.xpos,
                                 word.head,
                                 word.deprel,
                                 word.feats)]
                    df = pd.DataFrame(features, columns=["text", "lemma", "upos", "xpos", "head", "deprel", "feats"])
                    lst.append(df)
        tot_df = pd.concat(lst, ignore_index=True)
        tot_df = tot_df.shift(1).iloc[1:]
        tot_df["head"] = tot_df["head"].astype(int)
        return tot_df

    def print_stanza_analysis(self, text):
        text += " XX"
        doc = self.heb_nlp(text)
        lst = []
        for sen in doc.sentences:
            for token in sen.tokens:
                for word in token.words:
                    features = [(word.text,
                                 word.lemma,
                                 word.upos,
                                 word.xpos,
                                 word.head,
                                 word.deprel,
                                 word.feats)]

                    df = pd.DataFrame(features, columns=["text", "lemma", "upos", "xpos", "head", "deprel", "feats"])
                    lst.append(df)
        tot_df = pd.concat(lst, ignore_index=True)
        tot_df = tot_df.shift(1).iloc[1:]
        tot_df["head"] = tot_df["head"].astype(int)
        print(tot_df.head(50))


def clean_sentence(input_sentence):  # cleans urls, hashtags and tags
    cleaned_sentence = re.sub(r'http\S+', '', input_sentence)
    cleaned_sentence = re.sub(r"#", '', cleaned_sentence, flags=re.MULTILINE)
    cleaned_sentence = re.sub(r"@(\w+)", '', cleaned_sentence, flags=re.MULTILINE)
    return cleaned_sentence


def morphamize_sentence(input_sentence):
    processor = Processor()
    delimited = processor.get_analysis(input_sentence)
    text_array = delimited["text"].array
    morphamized_sentence = ' '.join(text_array).replace('_ ', " ").replace(' _', " ").replace("ל גליזציה", "לגליזציה")
    return morphamized_sentence

def create_morphamaized_file(dir, dest_dir, save=True):
    with open(dir) as fp:
        new_file_content = ""
        line = fp.readline()
        cnt = 1
        while line:
            sentence = line.split("\t")[0]
            cleaned_sentence = clean_sentence(sentence)
            morphamized_sentence = morphamize_sentence(cleaned_sentence)
            new_line = line.replace(sentence, morphamized_sentence)
            new_file_content += new_line
            line = fp.readline()
            cnt += 1
        fp.seek(0)
        fp.close()
        if save:
            f1 = open(dest_dir, 'a')
            f1.write(new_file_content)
            f1.seek(0)
            f1.close()


def cleanTxt(text):
    text = re.sub('@[A-Za-z0–9]+', '', text)  # Removing @mentions
    text = re.sub('#', '', text)  # Removing '#' hash tag
    text = re.sub('RT[\s]+', '', text)  # Removing RT
    text = re.sub('[a-zA-Z0-9]+(\.[a-zA-Z0-9]+)+', '', text)  # Removing hyperlink


    return text


def get_clean_data(df):
    assert "text" in df.keys()
    df['text'] = df['text'].apply(cleanTxt)
    return df

def morph_df(df,tosave=False, morph_path=""):

    morphamized_df = df
    morphamized_df['text'] = df['text'].apply(morphamize_sentence)
    if tosave:
        import io
        with io.open(morph_path, "w", encoding="utf-8") as f:
            f.write(morphamized_df.to_csv(encoding="utf-8", sep='\t', header=False, index=False))
    return morphamized_df


####################################### YAP #####################################################################
from Yap_Wrapper.yap_api import *
import pandas as pd
import re
from pathlib import Path


def yap_morphamize_sentence(input_sentence):
    ip = '127.0.0.1:8000'
    yap = YapApi()
    tokenized_text, segmented_text, lemmas, dep_tree, md_lattice, ma_lattice = yap.run(input_sentence, ip)
    return segmented_text

def yap_morph_df(df,tosave=False, morph_path=""):
    morphamized_df = df
    morphamized_df['text'] = df['text'].apply(yap_morphamize_sentence)
    if tosave:
        if os.path.exists(morph_path):
            os.remove(morph_path)
        import io
        with io.open(morph_path, "w", encoding="utf-8") as f:
            f.write(morphamized_df.to_csv(encoding="utf-8", sep='\t', header=False, index=False))
    return morphamized_df

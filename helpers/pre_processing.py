from helpers.json_to_spacy import json_to_spacy
from helpers.tsv_to_json import tsv_to_json


def pre_processing(df):
    print("\n> Pre-processing on data:")

    # remove useless columns
    df.drop(["id", "Doc_ID", "Sent_ID"], axis=1, inplace=True)

    df.to_csv('output/temp/train.tsv', sep='\t', encoding='utf-8', index=False)

    tsv_to_json("output/temp/train.tsv", 'output/temp/train.json', 'abc')

    json_to_spacy("output/temp/train.json", 'output/temp/train')

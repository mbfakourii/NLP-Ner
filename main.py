from builtins import print
from model.test import test_model
from model.spacy_ner_custom_entities import train_model
from pandas import read_csv
import spacy

if __name__ == "__main__":
    # print("\n> Concatenate datasets:")
    #
    # X_train = read_csv("files/train/train.csv", encoding="latin1")
    # print("\trows data train = " + len(X_train).__str__())
    #
    # # ------------- Save pre processing
    # aa = pre_processing(X_train)
    # exit(0)

    # ------------- Load File Pre processing
    # print("\n> Concatenate datasets:")
    # out = read_csv("output/temp/train")

    # ------------- Model
    nlp = train_model("en", 'new_model', "/home/mf/PycharmProjects/NER/output/model", 100)

    # ------------- Load Train Model
    # nlp = spacy.load("/home/mf/PycharmProjects/NER/output/model")
    #
    # # ------------- Test
    # X_test = read_csv("files/test/test.csv", encoding="latin1")
    # print("\trows data test = " + len(X_test).__str__())
    # test_model(nlp, X_test)

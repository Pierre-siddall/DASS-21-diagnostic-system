import warnings
import spacy
from csv import reader


def findSimilar(current):
    pass


def isNegative(word, negative):
    pass


def make_bow(words):
    bag = {}

    for word, tag in words:
        if word not in bag.keys():
            bag[word] = 1
        else:
            bag[word] += 1

    return bag


#
def extract_training_text(csvfile):
    training_text = []
    with open(csvfile, "r") as f:
        file_reader = reader(f)
        for i in file_reader:
            if i[-1] == '1':
                training_text.append(i)

    for text in training_text:
        text.pop(-1)

    return training_text


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    extract_training_text("sectraining.csv")

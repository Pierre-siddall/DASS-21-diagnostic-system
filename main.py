import warnings
import spacy
from csv import reader
from nltk.stem import WordNetLemmatizer


def make_bow(words):
    bag = {}

    for word, tag in words:
        if word not in bag.keys():
            bag[word] = 1
        else:
            bag[word] += 1

    return bag


# Function 3
def filter_tags(doc):
    whitelist = ['NOUN', 'ADJ', 'ADV', 'VERB']
    new_doc = []

    for token in doc:
        if token.pos_ in whitelist:
            new_doc.append(token)

    return new_doc


# Function 2
def tokenize_text(tokenizer, text):
    return tokenizer(text)


# Function 1
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


# Function 4
def get_ERT_emotions(csvfile):
    # TODO- Maybe extract the scores too?
    data = []
    with open(csvfile, "r") as f:
        file_reader = reader(f)
        for i in file_reader:
            data.append(i)

    considered = data[4:]
    emotions = []

    for record in considered:
        record_emotions = []
        for value in record[27:77:5]:
            record_emotions.append(value)
        emotions.append(record_emotions)

    for x in range(2):
        for emotion_list in emotions:
            if emotion_list[-1] == '':
                emotions.pop(emotions.index(emotion_list))

    return emotions


def generate_lexicon(emotions):
    lexicon = []
    wnl = WordNetLemmatizer()

    for emotion_record in emotions:
        for emotion in emotion_record:
            comparison_emotion = wnl.lemmatize(emotion.lower())
            if comparison_emotion not in lexicon:
                lexicon.append(comparison_emotion)

    return lexicon


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    nlp = spacy.load("en_core_web_sm")
    test = extract_training_text("sectraining.csv")
    doc = tokenize_text(nlp, test[0][0])
    new_doc = filter_tags(doc)
    ERT = get_ERT_emotions("ERT_dataset.csv")
    lex = generate_lexicon(ERT)


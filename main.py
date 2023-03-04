import warnings
import spacy
from csv import reader
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from sklearn.neural_network import MLPRegressor


# Function 9
def get_max_similarity(sims):
    max_score = -1
    max_text = ''

    for t, s in sims:
        if s > max_score:
            max_score = s
            max_text = t

    return max_score, max_text


# Function 8
def find_word_antonym(token):
    antonyms = []
    for syn in wordnet.synsets(token):
        for lem in syn.lemmas():
            if lem.antonyms():
                antonyms.append(lem.antonyms()[0].name())

    return antonyms[0]


# Function 7
def is_negated(token):
    if token.dep_ == 'neg':
        return True
    elif token.dep_ != 'neg':
        return False


# Function 6
def make_bow(words):
    bag = {}

    for word, tag in words:
        if word not in bag.keys():
            bag[word] = 1
        else:
            bag[word] += 1

    return bag


# Function 5
def discover_emotional_words(doc, lexicon, nlp_model, stopwords):
    discovered_words = []

    for token in doc:
        base = nlp_model(token.text)
        similarities = []
        for emotion in lexicon:

            comparison = nlp_model(emotion)
            negated = is_negated(token)

            if token.text not in stopwords:
                if negated:
                    new_comparison = nlp_model(find_word_antonym(comparison.text))
                    sim_score = base.similarity(new_comparison)
                    similarities.append((new_comparison.text, sim_score))
                else:
                    sim_score = base.similarity(comparison)
                    similarities.append((comparison.text, sim_score))

        score_max, text_max = get_max_similarity(similarities)

        if score_max >= 0.5:
            discovered_words.append(text_max)

        print(discovered_words)


# Function 3
def filter_tags(doc):
    # Cite for the whitelist
    whitelist = ['NOUN', 'ADJ', 'ADV', 'VERB']
    new_doc = []

    for token in doc:
        if token.pos_ in whitelist:
            new_doc.append(token)

    return new_doc


# Function 2
def tokenize_text(tokenizer, text):
    return tokenizer(text.lower())


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
    swords = stopwords.words('english')

    test = extract_training_text("sectraining.csv")
    doc = tokenize_text(nlp, test[0][0])
    new_doc = filter_tags(doc)
    ERT = get_ERT_emotions("ERT_dataset.csv")
    lex = generate_lexicon(ERT)

    discover_emotional_words(new_doc, lex, nlp, swords)

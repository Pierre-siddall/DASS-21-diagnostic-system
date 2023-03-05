import warnings
import spacy
import json
import numpy
import math
from csv import reader
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet


# TODO - Make unit tests, for calculate doc vector and find word antonym


# Function 11
def make_bow(discoveries):
    bow = {}

    for term in discoveries:
        if term not in bow.keys():
            bow[term] = 1
        elif term in bow.keys():
            bow[term] += 1

    return bow


# Function 10
def calculate_doc_vector(corpus, doc, lexicon, nlp_model, stopwords):
    vector = []

    with open("corpus_frequency.txt", "r") as f:
        text = f.read()
        frequencies = json.load(text)
        f.close()

    doc_discoveries = make_bow(discover_emotional_words(doc, lexicon, nlp_model, stopwords))

    for key in doc_discoveries.keys():
        tf_doc_term_freq = doc_discoveries[key]
        tf_doc_sum = sum(doc_discoveries.values())

        tf_value = tf_doc_term_freq / tf_doc_sum

        idf_num_docs = len(corpus)
        idf_appearances = frequencies[key]

        idf_value = math.log(idf_num_docs / idf_appearances)

        tfidf_value = tf_value * idf_value

        vector.append(tfidf_value)

    return vector


# Function 9
def get_corpus_frequency(corpus, lexicon, nlp_model, stopwords):
    all = {}
    count = 1

    for text in corpus:
        print(f"Reached text {count} out of {len(corpus)}")
        count += 1
        token_text = tokenize_text(nlp_model, text[0])
        discover_words = discover_emotional_words(token_text, lexicon, nlp_model, stopwords)
        word_set = set(discover_words)

        for word in word_set:
            if word not in all.keys():
                all[word] = 1
            elif word in all.keys():
                all[word] += 1

    with open("corpus_frequency.txt", "w") as f:
        f.write(json.dumps(all))
        f.close()


# Function 8
def get_max_similarity(sims):
    max_score = -1
    max_text = ''

    for t, s in sims:
        if s > max_score:
            max_score = s
            max_text = t

    return max_score, max_text


# Function 7
def find_word_antonym(token):
    antonyms = []
    for syn in wordnet.synsets(token):
        for lem in syn.lemmas():
            if lem.antonyms():
                antonyms.append(lem.antonyms()[0].name())

    if len(antonyms) == 0:
        return
    elif len(antonyms) > 0:
        return antonyms[0]


# Function 6
def is_negated(token):
    if token.dep_ == 'neg':
        return True
    elif token.dep_ != 'neg':
        return False


# Function 5
def discover_emotional_words(doc, lexicon, nlp_model, stopwords):
    discovered_words = []

    filtered_doc = filter_tags(doc)

    for token in filtered_doc:
        base = nlp_model(token.text)
        similarities = []
        for emotion in lexicon:

            comparison = nlp_model(emotion)
            negated = is_negated(token)

            try:
                if token.text not in stopwords and type(base) is not None and type(comparison) is not None:
                    if negated:
                        new_comparison = nlp_model(find_word_antonym(comparison.text))
                        sim_score = base.similarity(new_comparison)
                        similarities.append((new_comparison.text, sim_score))
                    else:
                        sim_score = base.similarity(comparison)
                        similarities.append((comparison.text, sim_score))
            except:
                pass

        score_max, text_max = get_max_similarity(similarities)

        if score_max >= 0.5:
            discovered_words.append(text_max)

    return discovered_words


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

    corpus = extract_training_text("sectraining.csv")
    doc = tokenize_text(nlp, corpus[0][0])
    ERT = get_ERT_emotions("ERT_dataset.csv")
    lex = generate_lexicon(ERT)

    discover_emotional_words(doc, lex, nlp, swords)

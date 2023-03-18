import warnings

import numpy as np
import pandas as pd
import spacy
import json
import math
from csv import reader
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from sklearn.neural_network import MLPRegressor


# TODO - Make unit tests, for calculate doc vector and find word antonym


################################  HELPER FUNCTIONS #################################################

#Function 14
def read_training_data(filename):

    traininglines=[]

    with open(filename,"r") as f:
        lines = f.read().splitlines()
        for line in lines:
            formattedline = line.split(' ')
            if formattedline[-1] == '-1':
                pass
            else:
                training_line_floats=[]
                for value in formattedline:
                    float_value = float(value)
                    training_line_floats.append(float_value)
                traininglines.append(training_line_floats)
    f.close()

    return pd.DataFrame(traininglines)



# Function 11
def make_bow(discoveries):
    bow = {}

    for term in discoveries:
        if term not in bow.keys():
            bow[term] = 1
        elif term in bow.keys():
            bow[term] += 1

    return bow


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


def generate_lexicon(emotions):
    # TODO - update this to reflect change to get_ERT_data function which adds scores
    lexicon = []
    wnl = WordNetLemmatizer()

    for emotion_record in emotions:
        for emotion in emotion_record[:-2]:
            comparison_emotion = wnl.lemmatize(emotion.lower())
            if comparison_emotion not in lexicon:
                lexicon.append(comparison_emotion)

    return lexicon


######################################################################################################

####################################### MAIN PROGRAM FUNCTIONS #######################################
# Function 13
def create_training_set(corpus, ERT_data, lexicon, nlp_model, stopwords):
    training_data = []

    print("Getting corpus data\n")
    converted, frequencies = get_corpus_data(corpus, lexicon, nlp_model, stopwords)

    with open("training_data.txt", "w") as t:
        t.close()

    with open("training_data.txt", "a") as f:
        print("Getting document vectors\n")
        for document in converted:
            document_vector = calculate_doc_vector(lexicon, ERT_data, corpus, document, frequencies)
            document_vector_string = " ".join(str(x) for x in document_vector)
            training_data.append(document_vector)
            f.write(document_vector_string + "\n")
        f.close()

    return training_data


# Function 12
def add_vector_target_output(ERT_data, doc_vector, doc_bow):
    doc_bow_list = []

    for key, value in doc_bow.items():
        doc_bow_list.append((value, key))

    sorted_vector = sorted(doc_bow_list)
    comparison_vector = np.array([t for v, t in sorted_vector[:-10:-1]])

    highest_similarity = 0
    highest_similarity_line = None

    for line in ERT_data:

        sorted_line = np.array(sorted(line[:10]))
        similarity = 0

        for element in comparison_vector:
            if element in sorted_line:
                similarity += 1

        if similarity > highest_similarity:
            highest_similarity = similarity
            highest_similarity_line = line

    if highest_similarity_line is not None:
        depression_score = highest_similarity_line[-2]
        anxiety_score = highest_similarity_line[-1]
    else:
        depression_score = -1
        anxiety_score = -1

    new_doc_vector = [v for v, t in doc_vector]
    new_doc_vector.append(depression_score)
    new_doc_vector.append(anxiety_score)

    return new_doc_vector


# Function 10
def calculate_doc_vector(lexicon, ERT_data, corpus, doc, corpus_frequency):
    vector = []

    doc_discoveries = make_bow(doc)

    for emotion in lexicon:
        if emotion in doc_discoveries.keys():
            tf_doc_term_freq = doc_discoveries[emotion]
            tf_doc_sum = sum(doc_discoveries.values())

            tf_value = tf_doc_term_freq / tf_doc_sum

            idf_num_docs = len(corpus)
            idf_appearances = corpus_frequency[emotion]

            idf_value = math.log(idf_num_docs / idf_appearances)

            tfidf_value = tf_value * idf_value

            vector.append((tfidf_value, emotion))
        else:
            vector.append((0.0, emotion))

    final_vector = add_vector_target_output(ERT_data, vector, doc_discoveries)

    return final_vector


# Function 9
def get_corpus_data(corpus, lexicon, nlp_model, stopwords):
    all = {}
    converted_docs = []
    count = 1

    for text in corpus:
        print(f"Reached text {count} out of {len(corpus)}")
        count += 1
        token_text = tokenize_text(nlp_model, text[0])
        discover_words = discover_emotional_words(token_text, lexicon, nlp_model, stopwords)
        converted_docs.append(discover_words)
        word_set = set(discover_words)

        for word in word_set:
            if word not in all.keys():
                all[word] = 1
            elif word in all.keys():
                all[word] += 1

    return converted_docs, all


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


# Function 1
def extract_training_text(csvfile):
    training_text = []
    with open(csvfile, "r") as f:
        file_reader = reader(f)
        for i in file_reader:
            training_text.append(i)

    for text in training_text:
        text.pop(-1)

    return training_text[::2]


# Function 4
def get_ERT_data(csvfile):
    data = []
    with open(csvfile, "r") as f:
        file_reader = reader(f)
        for i in file_reader:
            data.append(i)

    considered = data[4:]
    emotions_scores = []

    # Adding emotions_scores and scores
    for record in considered:
        record_values = []
        for emotion in record[27:77:5]:
            record_values.append(emotion)

        DASS_scores = record[153:174]
        record_depression = []
        record_anxiety = []
        scoring_template = ["S", "A", "D", "A", "D", "S", "A", "S", "A", "D", "S", "S", "D", "S",
                            "A", "D", "D", "S", "A", "A", "D"]

        # Adding DASS depression and anxiety scores
        for index in range(len(DASS_scores)):
            if scoring_template[index] == "A" and DASS_scores[index] != "46":
                try:
                    record_anxiety.append(int(DASS_scores[index]))
                except:
                    pass
            elif scoring_template[index] == "D" and DASS_scores[index] != "46":
                try:
                    record_depression.append(int(DASS_scores[index]))
                except:
                    pass

        record_values.append(sum(record_depression))
        record_values.append(sum(record_anxiety))
        emotions_scores.append(record_values)

    # Cleaning redundant records
    for x in range(4):
        for emotion_list in emotions_scores:
            if emotion_list[-1] == 0:
                emotions_scores.pop(emotions_scores.index(emotion_list))

    return emotions_scores


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    nlp = spacy.load("en_core_web_sm")
    swords = stopwords.words('english')

    corpus = extract_training_text("sectraining.csv")
    ERT = get_ERT_data("ERT_dataset.csv")
    lex = generate_lexicon(ERT)

    training_lines = read_training_data("training_data.txt")
    print(training_lines)



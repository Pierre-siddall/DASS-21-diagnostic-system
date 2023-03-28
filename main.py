import warnings
import numpy as np
import pandas as pd
import spacy
import math
from csv import reader
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler


################################  HELPER FUNCTIONS #################################################


def read_training_data(filename):
    traininglines = []

    with open(filename, "r") as f:
        lines = f.read().splitlines()
        for line in lines:
            formattedline = line.split(' ')
            if formattedline[-1] == '-1':
                pass
            else:
                training_line_values = []
                for value in formattedline:
                    if len(value) == 1 or len(value) == 2:
                        int_value = int(value)
                        training_line_values.append(int_value)
                    else:
                        float_value = float(value)
                        training_line_values.append(float_value)
                traininglines.append(training_line_values)
    f.close()

    return pd.DataFrame(traininglines)


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


def is_negated(token):
    if token.dep_ == 'neg':
        return True
    elif token.dep_ != 'neg':
        return False


def filter_tags(doc):
    # Cite for the whitelist
    whitelist = ['NOUN', 'ADJ', 'ADV', 'VERB']
    new_doc = []

    for token in doc:
        if token.pos_ in whitelist:
            new_doc.append(token)

    return new_doc


def tokenize_text(tokenizer, text):
    return tokenizer(text.lower())


def generate_lexicon(emotions):
    lexicon = []
    wnl = WordNetLemmatizer()

    for emotion_record in emotions:
        for emotion in emotion_record[:-2]:
            comparison_emotion = wnl.lemmatize(emotion.lower())
            if comparison_emotion not in lexicon:
                lexicon.append(comparison_emotion)

    return lexicon


def select_optimal_MLP_model(X_train, y_train):
    parameters = {
        "hidden_layer_sizes": [(100, 100, 100), (150, 150, 150), (200, 200, 200)],
        "activation": ["identity", "logistic", "tanh", "relu"], "solver": ["lbfgs", "sgd", "adam"],
        "learning_rate": ["constant", "invscaling", "adaptive"], "alpha": [0.0001, 0.0005, 0.001, 0.005],
    }

    clf = GridSearchCV(estimator=MLPRegressor(max_iter=10000), param_grid=parameters, cv=4, scoring="r2", n_jobs=10)
    clf.fit(X_train, y_train)

    return clf.best_params_


####################################### MAIN PROGRAM FUNCTIONS #######################################
def generate_dass_severity(depression_score, anxiety_score):
    if 0 <= depression_score <= 9:
        depression_level = "Normal"
    elif 10 <= depression_score <= 13:
        depression_level = "Mild"
    elif 14 <= depression_score <= 20:
        depression_level = "Moderate"
    elif 21 <= depression_score <= 27:
        depression_level = "Severe"
    elif 28 <= depression_score:
        depression_level = " Extremely severe"

    if 0 <= anxiety_score <= 7:
        anxiety_level = "Normal"
    elif 8 <= anxiety_score <= 9:
        anxiety_level = "Mild"
    elif 10 <= anxiety_score <= 14:
        anxiety_level = "Moderate"
    elif 15 <= anxiety_score <= 19:
        anxiety_level = "severe"
    elif 20 <= anxiety_score:
        anxiety_level = "Extremely severe"

    return depression_level, anxiety_level


def diagnose_document(filename, corpus, corpus_frequency, stopwords, ERT_data, lexicon, nlp_model, dataframe,
                      training_size):

    with open(filename,"r") as f:
        document_text = f.read()
    f.close()

    tokenized_doc = tokenize_text(nlp_model, document_text)
    emotion_doc = discover_emotional_words(tokenized_doc, lexicon, nlp_model, stopwords)
    document_vector = calculate_doc_vector(lexicon, ERT_data, corpus, emotion_doc, corpus_frequency,add_output=False)

    X = dataframe.loc[:, 0:210]
    y_depression = dataframe.loc[:, 211]
    y_anxiety = dataframe.loc[:, 212]

    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X, y_depression, train_size=training_size)
    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X, y_anxiety, train_size=training_size)

    # Scales the x axis data
    scaled_X = StandardScaler()
    X_training_scaled_d = scaled_X.fit_transform(X_train_d)
    X_training_scaled_a = scaled_X.fit_transform(X_train_a)

    regr_d = MLPRegressor(activation="identity",
                          alpha=0.001,
                          hidden_layer_sizes=(100, 100, 100),
                          learning_rate="adaptive",
                          solver="sgd",
                          max_iter=10000).fit(X_training_scaled_d, y_train_d)

    regr_a = MLPRegressor(activation="identity",
                          alpha=0.0001,
                          hidden_layer_sizes=(150, 150, 150),
                          learning_rate="invscaling",
                          solver="sgd",
                          max_iter=10000).fit(X_training_scaled_a, y_train_a)

    depression_score = regr_d.predict(document_vector)
    anxiety_score = regr_a.predict(document_vector)

    d_severity,a_severity = generate_dass_severity(depression_score,anxiety_score)

    return d_severity,a_severity




def validate_documents(labelled_corpus, training_data):
    depression_confirmed_proportions = {"Normal": 0, "Mild": 0, "Moderate": 0, "Severe": 0, "Extremely severe": 0}
    anxiety_confirmed_proportions = {"Normal": 0, "Mild": 0, "Moderate": 0, "Severe": 0, "Extremely severe": 0}
    depression_none_proportions = {"Normal": 0, "Mild": 0, "Moderate": 0, "Severe": 0, "Extremely severe": 0}
    anxiety_none_proportions = {"Normal": 0, "Mild": 0, "Moderate": 0, "Severe": 0, "Extremely severe": 0}

    for x in range(len(labelled_corpus)):
        if labelled_corpus[x][-1] == 1:

            depression_score = training_data[x][-2]
            anxiety_score = training_data[x][-1]
            d_class, a_class = generate_dass_severity(depression_score, anxiety_score)
            depression_confirmed_proportions[d_class] += 1
            anxiety_confirmed_proportions[a_class] += 1

        elif labelled_corpus[x][-1] == 0:
            depression_score = training_data[x][-2]
            anxiety_score = training_data[x][-1]
            d_class, a_class = generate_dass_severity(depression_score, anxiety_score)
            depression_none_proportions[d_class] += 1
            anxiety_none_proportions[a_class] += 1

    depression_confirmed_percentage = depression_confirmed_proportions["Moderate"] + \
                                      depression_confirmed_proportions["Severe"] + \
                                      depression_confirmed_proportions["Extremely severe"] \
                                      / sum(depression_confirmed_proportions.values())

    anxiety_confirmed_percentage = anxiety_confirmed_proportions["Moderate"] + \
                                   anxiety_confirmed_proportions["Severe"] + \
                                   anxiety_confirmed_proportions["Extremely severe"] \
                                   / sum(anxiety_confirmed_proportions.values())

    depression_none_percentage = depression_none_proportions["Moderate"] + \
                                 depression_none_proportions["Severe"] + \
                                 depression_none_proportions["Extremely severe"] \
                                 / sum(depression_none_proportions.values())

    anxiety_none_percentage = anxiety_none_proportions["Moderate"] + \
                              anxiety_none_proportions["Severe"] + \
                              anxiety_none_proportions["Extremely severe"] \
                              / sum(anxiety_none_proportions.values())

    return depression_confirmed_percentage, anxiety_confirmed_percentage, depression_none_percentage, \
        anxiety_none_percentage


def validate_MLP_regressor(dataframe, training_size, optimise=False):
    # Split the dataframe up into relevant columns
    X = dataframe.loc[:, 0:210]
    y_depression = dataframe.loc[:, 211]
    y_anxiety = dataframe.loc[:, 212]

    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X, y_depression, train_size=training_size)
    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X, y_anxiety, train_size=training_size)

    # Scales the x axis data
    scaled_X = StandardScaler()
    X_training_scaled_d = scaled_X.fit_transform(X_train_d)
    X_testing_scaled_d = scaled_X.fit_transform(X_test_d)

    X_training_scaled_a = scaled_X.fit_transform(X_train_a)
    X_testing_scaled_a = scaled_X.fit_transform(X_test_a)

    # Getting the optimal hyperparameters for the MLP
    if optimise:
        print("Getting optimised parameters for depression model")
        op_depression = select_optimal_MLP_model(X_training_scaled_d, y_train_d)
        print("Getting optimised parameters for anxiety model")
        op_anxiety = select_optimal_MLP_model(X_training_scaled_a, y_train_a)
        regr_d = MLPRegressor(activation=op_depression["activation"],
                              alpha=op_depression["alpha"],
                              hidden_layer_sizes=op_depression["hidden_layer_sizes"],
                              learning_rate=op_depression["learning_rate"],
                              solver=op_anxiety["solver"],
                              max_iter=10000).fit(X_training_scaled_d, y_train_d)

        regr_a = MLPRegressor(activation=op_anxiety["activation"],
                              alpha=op_anxiety["alpha"],
                              hidden_layer_sizes=op_anxiety["hidden_layer_sizes"],
                              learning_rate=op_anxiety["learning_rate"],
                              solver=op_anxiety["solver"],
                              max_iter=10000).fit(X_training_scaled_a, y_train_a)
    else:
        regr_d = MLPRegressor(activation="identity",
                              alpha=0.001,
                              hidden_layer_sizes=(100, 100, 100),
                              learning_rate="adaptive",
                              solver="sgd",
                              max_iter=10000).fit(X_training_scaled_d, y_train_d)

        regr_a = MLPRegressor(activation="identity",
                              alpha=0.0001,
                              hidden_layer_sizes=(150, 150, 150),
                              learning_rate="invscaling",
                              solver="sgd",
                              max_iter=10000).fit(X_training_scaled_a, y_train_a)

    regr_d.predict(X_testing_scaled_d)
    regr_a.predict(X_testing_scaled_a)

    print("The R squared score for the depression regressor is ", regr_d.score(X_testing_scaled_d, y_test_d))
    print("The R squared score for the anxiety regressor is ", regr_a.score(X_testing_scaled_a, y_test_a))


# Function 13
def create_training_set(corpus, ERT_data, lexicon, nlp_model, stopwords):
    training_data = []

    print("Getting corpus data\n")
    tokenized, frequencies = get_corpus_data(corpus, lexicon, nlp_model, stopwords)

    with open("training_data.txt", "w") as t:
        t.close()

    with open("training_data.txt", "a") as f:
        print("Getting document vectors\n")
        for document in tokenized:
            document_vector = calculate_doc_vector(lexicon, ERT_data, corpus, document, frequencies)
            document_vector_string = " ".join(str(x) for x in document_vector)
            training_data.append(document_vector)
            f.write(document_vector_string + "\n")
        f.close()

    return training_data


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


def calculate_doc_vector(lexicon, ERT_data, corpus, doc, corpus_frequency, add_output=True):
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

    if add_output:
        final_vector = add_vector_target_output(ERT_data, vector, doc_discoveries)
        return final_vector
    else:
        return vector


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


def extract_training_text(csvfile):
    training_text_labeled = []
    training_text = []
    with open(csvfile, "r") as f:
        file_reader = reader(f)
        for i in file_reader:
            training_text_labeled(i)
            training_text.append(i)

    for text in training_text:
        text.pop(-1)

    return training_text_labeled[::2], training_text[::2]


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


def main():
    warnings.filterwarnings("ignore")
    nlp = spacy.load("en_core_web_sm")
    swords = stopwords.words('english')
    corpus_labelled, corpus = extract_training_text("sectraining.csv")
    ERT = get_ERT_data("ERT_dataset.csv")
    lex = generate_lexicon(ERT)
    training_lines = read_training_data("training_data.txt")

    print("Welcome to the DASS-21 diagnostic system for authors of text\n")
    print("The options are:\n")
    print("(1) Validate\n")
    print("(2) Diagnose\n")
    print("(3) Exit\n")

    choice = int(input("Please enter a choice: "))

    while 0 > choice > 3 and type(choice) != int:
        choice = int(input("Please enter a choice between 1 and 3: "))

    if choice == 1:
        print("Here are the validation scores\n")
        dcp, acp, dnp, anp = validate_documents(corpus_labelled, training_lines)
        print(f"The training data which was confirmed to be depressed had {dcp} % of it's depression scores above the "
              f"threshold")
        print(f"The training data which was confirmed to be depressed had {acp} % of it's anxiety scores above the "
              f"threshold")
        print(f"The training data which was not confirmed to be depressed had {dnp} % of it's depression scores above "
              f"the threshold")
        print(f"The training data which was not confirmed to be depressed had {anp} % of it's anxiety scores above "
              f"the threshold")
        validate_MLP_regressor(training_lines, 0.7)
    elif choice == 2:
        pass
    elif choice == 3:
        exit()


if __name__ == '__main__':
    # Setup
    warnings.filterwarnings("ignore")
    nlp = spacy.load("en_core_web_sm")
    swords = stopwords.words('english')
    corpus_labelled, corpus = extract_training_text("sectraining.csv")
    ERT = get_ERT_data("ERT_dataset.csv")
    lex = generate_lexicon(ERT)

    training_lines = read_training_data("training_data.txt")
    validate_MLP_regressor(training_lines, 0.7)

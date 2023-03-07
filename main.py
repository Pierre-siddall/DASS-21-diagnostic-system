import warnings
import spacy
import json
import math
from csv import reader
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet


# TODO - Make unit tests, for calculate doc vector and find word antonym


################################  HELPER FUNCTIONS #################################################


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
# Function 12
def add_vector_target_output(ERT_data, doc_vector):
    sorted_vector = sorted(doc_vector)
    comparison_vector = sorted([t for v, t in sorted_vector[:10]])

    highest_similarity = 0
    highest_similarity_line = None

    for line in ERT_data:

        sorted_line = sorted(line[:10])
        similarity = 0

        for x in range(len(sorted_line)):
            if sorted_line[x] == comparison_vector[x]:
                similarity += 1

        if similarity > highest_similarity:
            highest_similarity = similarity
            highest_similarity_line = line

    depression_score = highest_similarity_line[-2]
    anxiety_score = highest_similarity_line[-1]

    new_doc_vector = [v for v, t in doc_vector]
    new_doc_vector.append(depression_score)
    new_doc_vector.append(anxiety_score)

    return new_doc_vector


# Function 10
def calculate_doc_vector(ERT_data, corpus, doc, lexicon, nlp_model, stopwords):
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

        vector.append((tfidf_value, key))

    final_vector = add_vector_target_output(ERT_data, vector)

    return final_vector


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
            if i[-1] == '1':
                training_text.append(i)

    for text in training_text:
        text.pop(-1)

    return training_text[:1000]


# Function 4
def get_ERT_data(csvfile):
    # TODO- Maybe extract the scores too?
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
    doc = tokenize_text(nlp, corpus[0][0])
    ERT = get_ERT_data("ERT_dataset.csv")
    lex = generate_lexicon(ERT)


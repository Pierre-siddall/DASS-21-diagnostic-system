from textblob import TextBlob


# Function 3
def make_bow(words):
    bag = {}

    for word, tag in words:
        if word not in bag.keys():
            bag[word]=1
        else:
            bag[word]+=1

    return bag



# Function 2
def filter_tags(blob):
    # TODO-Make in-place
    tags = blob.tags
    accepted = ['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNPS', 'RB', 'RBR', 'RBS', 'VB', 'VBD',
                'VBG', 'VBN', 'VBP', 'VBZ']

    filtered_tags = []
    for word, tag in tags:
        if tag in accepted:
            filtered_tags.append((word, tag))

    return filtered_tags


# Function 1
def blob_file(textfile):
    # TODO- potentially check to see if file format affects speed
    # Done for a general file set
    with open(textfile, "r") as f:
        text = f.read()
    blob_object = TextBlob(text)

    return blob_object


if __name__ == '__main__':
    my_blob = blob_file("Note.txt")
    tags = filter_tags(my_blob)
    testBag= make_bow(tags)
    print(testBag)

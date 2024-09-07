import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to map NLTK POS tags to WordNet POS tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun if no match

# Function to test preprocessing on a random string
def Test():
    test = 'Thi s is a TeSTING str ing, 5'
    print('\nOriginal string: ' + test)

    stop_words = set(stopwords.words('english'))
    preProcessedTest = PreprocessText(test, stop_words)
    print('Preprocessed: ', preProcessedTest)

# Function that will preprocess the text, remove stopwords, and lemmatize
def PreprocessText(text, stop_words):
    # Remove the blank spaces in the text and return text without whitespace.
    text = re.sub(r'\s+', ' ', text).strip()

    # Makes the text lowercase.
    text = text.lower()

    # Tokenize the data using nltk's word_tokenize.
    tokenizedData = word_tokenize(text)

    # Remove stopwords from the tokenized data.
    tokenizedData = [word for word in tokenizedData if word not in stop_words]

    # Get POS tags for each token
    pos_tags = nltk.pos_tag(tokenizedData)

    # Apply lemmatization with the correct POS tag
    tokenizedData = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]

    return tokenizedData  # Return the lemmatized data as a list

def main():
    # Load stopwords from nltk
    stop_words = set(stopwords.words('english'))

    Test()

    # Read the provided CSV files
    fullTrainingSet = pd.read_csv('BBC_train_full.csv')
    testSet = pd.read_csv('test_data.csv')

    # Apply preprocessing (including stopword removal and lemmatization) to the training and testing sets
    fullTrainingSet['text'] = fullTrainingSet['text'].apply(lambda text: PreprocessText(text, stop_words))
    testSet['text'] = testSet['text'].apply(lambda text: PreprocessText(text, stop_words))

    # Save the preprocessed data to new CSV files
    fullTrainingSet.to_csv('BBC_train_full_preprocessed_with_lemmatization.csv', index=False)
    testSet.to_csv('test_data_preprocessed_with_lemmatization.csv', index=False)

    print('\nSaved the training and test sets with stopwords removed and lemmatized.\n')

if __name__ == '__main__':
    main()
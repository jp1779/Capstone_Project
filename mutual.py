import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Function called just to see if the other functions work on a random string.
def Test():
    test = 'Thi s is a TeSTING str ing, 5'
    print('\nOriginal string: ' + test)

    stop_words = set(stopwords.words('english'))
    preProcessedTest = PreprocessText(test, stop_words)
    print('Preprocessed: ', preProcessedTest)

# Function that will preprocess the text and remove stopwords.
def PreprocessText(text, stop_words):
    # Remove the blank spaces in the text and return text without whitespace.
    text = re.sub(r'\s+', ' ', text).strip()

    # Makes the text lowercase.
    text = text.lower()

    # Tokenize the data using nltk's word_tokenize.
    tokenizedData = word_tokenize(text)

    # Remove stopwords from the tokenized data.
    tokenizedData = [word for word in tokenizedData if word not in stop_words]

    return tokenizedData  # Return the tokenized data as a list (not a string).

def main():
    # Load stopwords from nltk.
    stop_words = set(stopwords.words('english'))

    Test()

    # Read the provided CSV files.
    fullTrainingSet = pd.read_csv('BBC_train_full.csv')
    testSet = pd.read_csv('test_data.csv')

    # Apply preprocessing (including stopword removal) to the training and testing sets.
    fullTrainingSet['text'] = fullTrainingSet['text'].apply(lambda text: PreprocessText(text, stop_words))
    testSet['text'] = testSet['text'].apply(lambda text: PreprocessText(text, stop_words))

    # Save the preprocessed data to new CSV files.
    fullTrainingSet.to_csv('BBC_train_full_preprocessed_with_stopwords.csv', index=False)
    testSet.to_csv('test_data_preprocessed_with_stopwords.csv', index=False)

    print('\nSaved the training and test sets with stopwords removed and tokenized as lists\n')

if __name__ == '__main__':
    main()

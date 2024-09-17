import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

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

# Function that will preprocess the data so that the ML models can use it.
# Removes numbers, removes symbols, makes all text lowercase,
# tokenizes the text, removes stopwords, and applies lemmatization. 
def PreprocessText(text):

    # Load stopwords from nltk
    stop_words = set(stopwords.words('english'))

    # Remove numbers and symbols from the data.
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove consecutive blank spaces between words and trailing/leading whitspace.
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

# Function that uses the Naive Bayes model for news classification.
def naiveBayes(trainingSet, testSet, testLabels):
    
    # Obtain the data that corresponds to each labeled column in the sets.
    trainingText = trainingSet['text']
    trainingCategory = trainingSet['category']
    testText = testSet['text']

    # We currently have our sets tokenized. We must convert them back to a single string
    # for the count vectorizer to be used.
    trainingText = trainingText.apply(lambda tokens: ' '.join(tokens))
    testText = testText.apply(lambda tokens: ' '.join(tokens))

    # Count vectorizers converts the string to numerical format. This
    # is the Bag of Words model that will be used for Naive Bayes classification.
    # It uses term frequency rather than word order and context.
    # fit transform learns the vocab of the training set and transforms into a matrix
    # of word frequency.
    vectorizer = CountVectorizer()
    trainingTextVector = vectorizer.fit_transform(trainingText)
    testTextVector = vectorizer.transform(testText) 

    # Train the Multinomial Naive Bayes model. It expects word counts which
    # we already have thanks to the count vectorizer. We will train the model
    # using the vectorized training set and its categories. It estimates
    # the likelihood of each category given the word frequencies.
    naiveClassifier = MultinomialNB()
    naiveClassifier.fit(trainingTextVector, trainingCategory)

    # Now predict the categories of the news articles in the test set by
    # using their word frequencies (vectorized test set).
    categoryPredictions = naiveClassifier.predict(testTextVector)

    # Evaluate the classifications by comparing the predictions to
    # the actual labels data set we were provided. The classification
    # report will give the precision, recall, and F1-score.
    accuracy = accuracy_score(testLabels, categoryPredictions)
    report = classification_report(testLabels, categoryPredictions, target_names = ['0 (Business)', '1 (Entertainment)', ' 2(Politics)', '3 (Sport)', '4 (Tech)'])

    # Print the info. Accuracy is formatted to two decimal places.
    print('\nNaive Bayes Accuracy: {:.2f}%'.format(accuracy * 100))
    print('\nNaive Bayes Classification Report:\n', report)

def main():

    # Read the provided CSV files.
    fullTrainingSet = pd.read_csv('BBC_train_full.csv')
    testSet = pd.read_csv('test_data.csv')
    testLabels = pd.read_csv('test_labels.csv')

    # Apply preprocessing techniques to the training and testing sets.
    fullTrainingSet['text'] = fullTrainingSet['text'].apply(lambda text: PreprocessText(text))
    testSet['text'] = testSet['text'].apply(lambda text: PreprocessText(text))

    # Split the full training set into 3 equal-sized subsets.
    # Full Entries: 1726. T1 Entries: 576. T2 Entries: 575. T3 Entries: 575.
    trainingSet1, trainingSet2, trainingSet3 = np.array_split(fullTrainingSet, 3)
    trainingSet3RemovedLabels = trainingSet3.drop(columns = ['category'])
    trainingSet3Labels = trainingSet3['category']

    # Save the preprocessed data to new CSV files.
    fullTrainingSet.to_csv('BBC_train_full_preprocessed.csv', index=False)
    testSet.to_csv('test_data_preprocessed.csv', index=False)
    print('\nSuccessfuly preprocessed the data.\n')

    ### Uncomment the 4 lines below to check if the data was split properly (it is).
    #trainingSet1.to_csv('train1Check.csv', index=False)
    #trainingSet2.to_csv('train2Check.csv', index=False)
    #trainingSet3RemovedLabels.to_csv('train3LabelessCheck.csv', index=False)
    #trainingSet3Labels.to_csv('train3LabelsOnlyCheck.csv', index=False)

    # Use the Naive Bayes model and evaluate.
    naiveBayes(fullTrainingSet, testSet, testLabels['category'])

    

if __name__ == '__main__':
    main()
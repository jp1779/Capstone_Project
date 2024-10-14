import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Initialize the lemmatizer and stemmer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

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

# Function to preprocess text
def PreprocessText(text):
    # Load stopwords from nltk
    stop_words = set(stopwords.words('english'))

    # Remove numbers and symbols from the data
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove consecutive blank spaces and make text lowercase
    text = re.sub(r'\s+', ' ', text).strip().lower()

    # Tokenize the data
    tokenizedData = word_tokenize(text)

    # Remove stopwords
    tokenizedData = [word for word in tokenizedData if word not in stop_words]

    # Get POS tags
    pos_tags = nltk.pos_tag(tokenizedData)

    # Apply stemming and lemmatization
    tokenizedData = [stemmer.stem(word) for word in tokenizedData]
    tokenizedData = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]

    return tokenizedData  # Return the processed data as a list

# Naive Bayes Model function
def naiveBayes(trainingSet, testSet, testLabels):
    # Obtain the data
    trainingText = trainingSet['text']
    trainingCategory = trainingSet['category']
    testText = testSet['text']

    # Vectorize the text
    vectorizer = CountVectorizer()
    trainingTextVector = vectorizer.fit_transform(trainingText)
    testTextVector = vectorizer.transform(testText)

    # Train the Naive Bayes model
    naiveClassifier = MultinomialNB()
    naiveClassifier.fit(trainingTextVector, trainingCategory)

    # Predict and evaluate
    categoryPredictions = naiveClassifier.predict(testTextVector)
    accuracy = accuracy_score(testLabels, categoryPredictions)
    report = classification_report(testLabels, categoryPredictions,
                                   target_names=['0 (Business)', '1 (Entertainment)',
                                                 '2 (Politics)', '3 (Sport)', '4 (Tech)'])

    # Print results
    print(f'\nNaive Bayes Accuracy: {accuracy * 100:.2f}%')
    print(f'\nNaive Bayes Classification Report:\n{report}')

    return naiveClassifier, vectorizer, accuracy, report

# Function to predict labels and posterior probabilities using Naive Bayes
def predict_with_probabilities_nb(model, vectorizer, textData):
    textVector = vectorizer.transform(textData)
    predicted_labels = model.predict(textVector)
    predicted_probabilities = model.predict_proba(textVector)
    return predicted_labels, predicted_probabilities

def main():
    # Read the provided CSV files
    fullTrainingSet = pd.read_csv('BBC_train_full.csv')
    testSet = pd.read_csv('test_data.csv')
    testLabels = pd.read_csv('test_labels.csv')

    # Apply preprocessing to the training and testing sets
    fullTrainingSet['text'] = fullTrainingSet['text'].apply(lambda text: PreprocessText(text))
    testSet['text'] = testSet['text'].apply(lambda text: PreprocessText(text))

    # Convert tokenized text back to a single string
    fullTrainingSet['text'] = fullTrainingSet['text'].apply(lambda tokens: ' '.join(tokens))
    testSet['text'] = testSet['text'].apply(lambda tokens: ' '.join(tokens))

    # Split the full training set into 3 equal-sized subsets
    trainingSet1, trainingSet2, trainingSet3 = np.array_split(fullTrainingSet, 3)

    # Part 1: Train Naive Bayes models on trainingSet1 and trainingSet2
    print("\nTraining Naive Bayes model on Training Set 1")
    model1, vectorizer1, accuracy1, report1 = naiveBayes(trainingSet1, testSet, testLabels['category'])

    print("\nTraining Naive Bayes model on Training Set 2")
    model2, vectorizer2, accuracy2, report2 = naiveBayes(trainingSet2, testSet, testLabels['category'])

    # Part 2: Mutual Learning between the two Naive Bayes models
    # Remove labels from trainingSet3
    trainingSet3NoLabels = trainingSet3.drop(columns=['category'])

    # Use both models to predict labels and probabilities for trainingSet3
    labels1, probs1 = predict_with_probabilities_nb(model1, vectorizer1, trainingSet3NoLabels['text'])
    labels2, probs2 = predict_with_probabilities_nb(model2, vectorizer2, trainingSet3NoLabels['text'])

    # For each data point, choose the label from the model with the highest confidence
    new_labels = []
    for i in range(len(trainingSet3NoLabels)):
        max_prob1 = max(probs1[i])
        max_prob2 = max(probs2[i])
        if max_prob1 >= max_prob2:
            new_labels.append(labels1[i])
        else:
            new_labels.append(labels2[i])

    # Add the new labels to trainingSet3
    trainingSet3NoLabels['category'] = new_labels

    # Retrain both models using their original training sets plus the newly labeled trainingSet3
    combinedSet1 = pd.concat([trainingSet1, trainingSet3NoLabels])
    combinedSet2 = pd.concat([trainingSet2, trainingSet3NoLabels])

    print("\nRetraining Naive Bayes model 1 with Training Set 1 and newly labeled Training Set 3")
    model1_retrained, vectorizer1_retrained, accuracy1_retrained, report1_retrained = naiveBayes(
        combinedSet1, testSet, testLabels['category'])

    print("\nRetraining Naive Bayes model 2 with Training Set 2 and newly labeled Training Set 3")
    model2_retrained, vectorizer2_retrained, accuracy2_retrained, report2_retrained = naiveBayes(
        combinedSet2, testSet, testLabels['category'])

    # Report the initial and retrained accuracies
    print(f"\nInitial Naive Bayes Model 1 Accuracy: {accuracy1 * 100:.2f}%")
    print(f"Retrained Naive Bayes Model 1 Accuracy: {accuracy1_retrained * 100:.2f}%")

    print(f"\nInitial Naive Bayes Model 2 Accuracy: {accuracy2 * 100:.2f}%")
    print(f"Retrained Naive Bayes Model 2 Accuracy: {accuracy2_retrained * 100:.2f}%")

if __name__ == '__main__':
    main()

import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
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

# Function to predict labels and return probabilities
def predict_with_probabilities(svmModel, vectorizedText):
    # We are using CalibratedClassifierCV to get probability estimates
    calibratedModel = CalibratedClassifierCV(svmModel)
    calibratedModel.fit(vectorizedText, svmModel.predict(vectorizedText))
    return calibratedModel.predict(vectorizedText), calibratedModel.predict_proba(vectorizedText)

# Function to relabel Training Set 3 based on highest confidence from both SVM models
def relabelTrainingSet3(trainingSet1, trainingSet3RemovedLabels, vectorizer):
    # Train Linear SVM on Training Set 1
    linearSVM = SVC(kernel='linear', probability=True)
    trainingTextVector = vectorizer.transform(trainingSet1['text'])
    linearSVM.fit(trainingTextVector, trainingSet1['category'])

    # Train Non-Linear SVM (Sigmoid Kernel) on Training Set 1
    nonLinearSVM = SVC(kernel='sigmoid', probability=True)
    nonLinearSVM.fit(trainingTextVector, trainingSet1['category'])

    # Vectorize the text from Training Set 3
    trainingSet3Vector = vectorizer.transform(trainingSet3RemovedLabels['text'])

    # Get predictions and probabilities from both models
    linear_predictions, linear_probabilities = predict_with_probabilities(linearSVM, trainingSet3Vector)
    non_linear_predictions, non_linear_probabilities = predict_with_probabilities(nonLinearSVM, trainingSet3Vector)

    # Relabel Training Set 3 based on the model with the highest confidence
    new_labels = []
    for i in range(len(trainingSet3RemovedLabels)):
        if max(linear_probabilities[i]) > max(non_linear_probabilities[i]):
            new_labels.append(linear_predictions[i])  # Use label from linear SVM
        else:
            new_labels.append(non_linear_predictions[i])  # Use label from non-linear SVM

    # Add the new labels to Training Set 3 and return
    trainingSet3RemovedLabels['category'] = new_labels
    return trainingSet3RemovedLabels

# Function to retrain and evaluate the SVM models
def retrainAndEvaluate(trainingSet1, trainingSet3Relabeled, testSet, testLabels):
    # Combine Training Set 1 and the newly labeled Training Set 3
    combinedTrainingSet = pd.concat([trainingSet1, trainingSet3Relabeled])

    # Vectorize the combined dataset
    vectorizer = CountVectorizer()
    combinedTextVector = vectorizer.fit_transform(combinedTrainingSet['text'])
    testTextVector = vectorizer.transform(testSet['text'])

    # Retrain Linear SVM
    linearSVM = SVC(kernel='linear')
    linearSVM.fit(combinedTextVector, combinedTrainingSet['category'])
    linear_predictions = linearSVM.predict(testTextVector)

    # Retrain Non-Linear SVM (Sigmoid Kernel)
    nonLinearSVM = SVC(kernel='sigmoid')
    nonLinearSVM.fit(combinedTextVector, combinedTrainingSet['category'])
    non_linear_predictions = nonLinearSVM.predict(testTextVector)

    # Evaluate both models
    linear_accuracy = accuracy_score(testLabels, linear_predictions)
    non_linear_accuracy = accuracy_score(testLabels, non_linear_predictions)

    linear_report = classification_report(testLabels, linear_predictions, target_names=['0 (Business)', '1 (Entertainment)', '2 (Politics)', '3 (Sport)', '4 (Tech)'])
    non_linear_report = classification_report(testLabels, non_linear_predictions, target_names=['0 (Business)', '1 (Entertainment)', '2 (Politics)', '3 (Sport)', '4 (Tech)'])

    # Print results
    print(f'\nRetrained Linear SVM Accuracy: {linear_accuracy * 100:.2f}%')
    print(f'Retrained Linear SVM Classification Report:\n{linear_report}')

    print(f'\nRetrained Non-Linear SVM (Sigmoid) Accuracy: {non_linear_accuracy * 100:.2f}%')
    print(f'Retrained Non-Linear SVM Classification Report:\n{non_linear_report}')

    return linear_accuracy, non_linear_accuracy

def main():

    # Read the provided CSV files.
    fullTrainingSet = pd.read_csv('BBC_train_full.csv')
    testSet = pd.read_csv('test_data.csv')
    testLabels = pd.read_csv('test_labels.csv')

    # Apply preprocessing techniques to the training and testing sets.
    fullTrainingSet['text'] = fullTrainingSet['text'].apply(lambda text: PreprocessText(text))
    testSet['text'] = testSet['text'].apply(lambda text: PreprocessText(text))

    # Convert tokenized text back to a single string (join tokens)
    fullTrainingSet['text'] = fullTrainingSet['text'].apply(lambda tokens: ' '.join(tokens))
    testSet['text'] = testSet['text'].apply(lambda tokens: ' '.join(tokens))

    # Split the full training set into 3 equal-sized subsets.
    trainingSet1, trainingSet2, trainingSet3 = np.array_split(fullTrainingSet, 3)
    trainingSet3RemovedLabels = trainingSet3.drop(columns=['category'])  # Ensure labels are removed

    # Save the preprocessed data to new CSV files.
    fullTrainingSet.to_csv('BBC_train_full_preprocessed.csv', index=False)
    testSet.to_csv('test_data_preprocessed.csv', index=False)
    print('\nSuccessfully preprocessed the data.\n')

    # Vectorize the text in trainingSet1
    vectorizer = CountVectorizer()
    vectorizer.fit(trainingSet1['text'])

    # Relabel Training Set 3 based on SVM confidence
    relabeled_trainingSet3 = relabelTrainingSet3(trainingSet1, trainingSet3RemovedLabels, vectorizer)
    print('\nSuccessfully relabeled Training Set 3.\n')

    # Retrain both SVM models and evaluate them on the test set
    linear_accuracy, non_linear_accuracy = retrainAndEvaluate(trainingSet1, relabeled_trainingSet3, testSet, testLabels['category'])

    # Print the results in a table format
    print("\nRetrained SVM Model Accuracies:")
    print(f"Linear SVM: {linear_accuracy * 100:.2f}%")
    print(f"Non-linear SVM (Sigmoid Kernel): {non_linear_accuracy * 100:.2f}%")

if __name__ == '__main__':
    main()

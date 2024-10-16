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
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

# Initialize the stemmer and lemmatizer
stemmer = PorterStemmer()
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
def PreprocessText(text):

    stop_words = set(stopwords.words('english')) # Load stopwords from nltk

    text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove numbers and symbols from the data.
    text = re.sub(r'\s+', ' ', text).strip() # Remove spaces
    text = text.lower() # Makes the text lowercase.

    tokenizedData = word_tokenize(text) # Tokenize
    tokenizedData = [word for word in tokenizedData if word not in stop_words] # Remove stopwords
    pos_tags = nltk.pos_tag(tokenizedData) # Get POS tags for each token

    tokenizedData = [stemmer.stem(word, get_wordnet_pos(tag)) for word, tag in pos_tags] # Apply stemming
    tokenizedData = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags] # Apply lemmatization

    return tokenizedData  # Return the lemmatized data as a list

# Naive Bayes Model Full Training
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
    report = classification_report(testLabels, categoryPredictions, target_names=['0 (Business)', '1 (Entertainment)', '2 (Politics)', '3 (Sport)', '4 (Tech)'])

    # Print results
    print(f'\nNaive Bayes Accuracy: {accuracy * 100:.2f}%')
    print(f'\nNaive Bayes Classification Report:\n{report}')

# MLP Model Full Training
def neuralNetwork(trainingSet, testSet, testLabels):

    # Obtain the data
    trainingText = trainingSet['text']
    trainingCategory = trainingSet['category']
    testText = testSet['text']

    # Vectorize the text
    vectorizer = CountVectorizer()
    trainingTextVector = vectorizer.fit_transform(trainingText)
    testTextVector = vectorizer.transform(testText)

    # Train the MLP Neural Network
    mlpModel = MLPClassifier(hidden_layer_sizes = (100, 100), max_iter = 500, random_state = 1)
    mlpModel.fit(trainingTextVector, trainingCategory)

    # Predict and evaluate
    categoryPredictions = mlpModel.predict(testTextVector)
    accuracy = accuracy_score(testLabels, categoryPredictions)
    report = classification_report(testLabels, categoryPredictions, target_names=['0 (Business)', '1 (Entertainment)', '2 (Politics)', '3 (Sport)', '4 (Tech)'])

    # Print results
    print(f'\nMLP Neural Network Accuracy: {accuracy * 100:.2f}%')
    print(f'\nMLP Neural Network Classification Report:\n{report}')

# Homogeneous MLP NN Training
def mutalNeuralNetwork(trainingSet1, trainingSet2, trainingSet3, testSet, testLabels):
    
    # Vectorize the text
    vectorizer = CountVectorizer()
    trainingTextVector1 = vectorizer.fit_transform(trainingSet1['text'])
    trainingTextVector2 = vectorizer.transform(trainingSet2['text'])
    trainingTextVector3 = vectorizer.transform(trainingSet3['text'])  # For mutual learning
    
    # Prepare test data
    testTextVector = vectorizer.transform(testSet['text'])

    # Train two separate MLP models (NN1 and NN2) on trainingSet1 and trainingSet2
    mlpModel1 = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=500, random_state=1)
    mlpModel1.fit(trainingTextVector1, trainingSet1['category'])

    mlpModel2 = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=500, random_state=2)
    mlpModel2.fit(trainingTextVector2, trainingSet2['category'])

    # Test both models on the test set before mutual learning
    predictions1_initial = mlpModel1.predict(testTextVector)
    predictions2_initial = mlpModel2.predict(testTextVector)

    accuracy1_initial = accuracy_score(testLabels, predictions1_initial)
    accuracy2_initial = accuracy_score(testLabels, predictions2_initial)

    print(f'\nInitial Accuracy of NN1 on test set (using trainingSet1): {accuracy1_initial * 100:.2f}%')
    print(f'\nInitial Accuracy of NN2 on test set (using trainingSet2): {accuracy2_initial * 100:.2f}%')

    # Predict the labels and probabilities for trainingSet3
    predictions1 = mlpModel1.predict(trainingTextVector3)
    probabilities1 = mlpModel1.predict_proba(trainingTextVector3)

    predictions2 = mlpModel2.predict(trainingTextVector3)
    probabilities2 = mlpModel2.predict_proba(trainingTextVector3)

    # Relabel trainingSet3 based on the highest confidence from the two NNs
    new_labels = []
    for i in range(len(trainingSet3)):
        if max(probabilities1[i]) > max(probabilities2[i]):
            new_labels.append(predictions1[i])
        else:
            new_labels.append(predictions2[i])

    # Create a new DataFrame for trainingSet3 with the new labels
    trainingSet3['category'] = new_labels

    # Retrain both NNs using trainingSet1 + relabeled trainingSet3, and trainingSet2 + relabeled trainingSet3
    combinedTrainingSet1 = pd.concat([trainingSet1, trainingSet3])
    combinedTrainingSet2 = pd.concat([trainingSet2, trainingSet3])

    combinedTextVector1 = vectorizer.transform(combinedTrainingSet1['text'])
    combinedTextVector2 = vectorizer.transform(combinedTrainingSet2['text'])

    # Retrain the models
    mlpModel1.fit(combinedTextVector1, combinedTrainingSet1['category'])
    mlpModel2.fit(combinedTextVector2, combinedTrainingSet2['category'])

    # Test both models on the test set
    predictions1 = mlpModel1.predict(testTextVector)
    predictions2 = mlpModel2.predict(testTextVector)

    accuracy1 = accuracy_score(testLabels, predictions1)
    accuracy2 = accuracy_score(testLabels, predictions2)

    print(f'\nHomogenous Mutual Learning NN1 Accuracy: {accuracy1 * 100:.2f}%')
    print(f'\nHomogenous Mutual Learning NN2 Accuracy: {accuracy2 * 100:.2f}%')

    report1 = classification_report(testLabels, predictions1, target_names=['0 (Business)', '1 (Entertainment)', '2 (Politics)', '3 (Sport)', '4 (Tech)'])
    report2 = classification_report(testLabels, predictions2, target_names=['0 (Business)', '1 (Entertainment)', '2 (Politics)', '3 (Sport)', '4 (Tech)'])

    print(f'\nHomogenous Mutual Learning NN1 Classification Report:\n{report1}')
    print(f'\nHomogenous Mutual Learning NN2 Classification Report:\n{report2}')
   

# SVM models Full Training
def trainSVM(trainingSet, testSet, testLabels, kernel_type):

    # Obtain the data.
    trainingText = trainingSet['text']
    trainingCategory = trainingSet['category']
    testText = testSet['text']

    # Learn vocab and transform to word count vector.
    vectorizer = CountVectorizer()
    trainingTextVector = vectorizer.fit_transform(trainingText)
    testTextVector = vectorizer.transform(testText)

    # Train the SVM model
    svmModel = SVC(kernel=kernel_type)
    svmModel.fit(trainingTextVector, trainingCategory)
    
    # Predict the categories of the news.
    categoryPredictions = svmModel.predict(testTextVector)

    # Evaluate the performance
    accuracy = accuracy_score(testLabels, categoryPredictions)
    report = classification_report(testLabels, categoryPredictions,
                                   target_names=['0 (Business)', '1 (Entertainment)', '2 (Politics)', '3 (Sport)',
                                                 '4 (Tech)'])
    # Print the info
    print(f'\nSVM ({kernel_type}) Accuracy: {accuracy * 100:.2f}%')
    print(f'\nSVM ({kernel_type}) Classification Report:\n', report)

    return accuracy

# SVM Function to predict labels and return probabilities
def predict_with_probabilities(svmModel, vectorizedText):
    # Use CalibratedClassifierCV to get probability estimates
    calibratedModel = CalibratedClassifierCV(svmModel)
    calibratedModel.fit(vectorizedText, svmModel.predict(vectorizedText))
    return calibratedModel.predict(vectorizedText), calibratedModel.predict_proba(vectorizedText)

# SVM Function to relabel Training Set 3 based on highest confidence from both SVM models
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

# SVM Function to retrain and evaluate the SVM models
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

    # Full Naive Bayes Model
    naiveBayes(fullTrainingSet, testSet, testLabels['category'])

    # Full MLP Neural Network Model
    neuralNetwork(fullTrainingSet, testSet, testLabels['category'])

    # Full SVM models
    trainSVM(fullTrainingSet, testSet, testLabels['category'], kernel_type='linear')
    trainSVM(fullTrainingSet, testSet, testLabels['category'], kernel_type='sigmoid')

    mutalNeuralNetwork(trainingSet1, trainingSet2, trainingSet3RemovedLabels, testSet, testLabels['category'])

 

    # Vectorize the text in trainingSet1
    #vectorizer = CountVectorizer()
    #vectorizer.fit(trainingSet1['text'])

    # Relabel Training Set 3 based on SVM confidence
    #relabeled_trainingSet3 = relabelTrainingSet3(trainingSet1, trainingSet3RemovedLabels, vectorizer)
    #print('\nSuccessfully relabeled Training Set 3.\n')

    # Retrain both SVM models and evaluate them on the test set
    #linear_accuracy, non_linear_accuracy = retrainAndEvaluate(trainingSet1, relabeled_trainingSet3, testSet, testLabels['category'])

    # Print the results in a table format
    #print("\nRetrained SVM Model Accuracies:")
    #print(f"Linear SVM: {linear_accuracy * 100:.2f}%")
    #print(f"Non-linear SVM (Sigmoid Kernel): {non_linear_accuracy * 100:.2f}%")

if __name__ == '__main__':
    main()

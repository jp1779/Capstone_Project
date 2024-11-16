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

from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

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

# Naive Bayes Model function
def naiveBayesIsrael(trainingSet, testSet, testLabels):
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

# Ensemble method to combine model predictions using soft voting
def ensemble_soft_voting(probs1, probs2, classes_):
    avg_probs = (probs1 + probs2) / 2  # Average the probabilities from both models
    ensemble_label_indices = avg_probs.argmax(axis=1)  # Choose the label with the highest average probability
    ensemble_labels = [classes_[i] for i in ensemble_label_indices]  # Convert indices back to class labels
    return ensemble_labels

# Main function with soft voting ensemble method
def mutualNaiveBayes(trainingSet1, trainingSet2, trainingSet3, testSet, testLabels):

    # Part 1: Train Naive Bayes models on trainingSet1 and trainingSet2
    print("\nInitial Accuracy of Naive Bayes Model 1 on Training Set 1")
    model1, vectorizer1, accuracy1_before, report1 = naiveBayesIsrael(trainingSet1, testSet, testLabels['category'])

    print("\nInitial Accuracy of Naive Bayes Model 2 on Training Set 2")
    model2, vectorizer2, accuracy2_before, report2 = naiveBayesIsrael(trainingSet2, testSet, testLabels['category'])

    # Part 2: Evaluate ensemble model before retraining
    _, probs1_test = predict_with_probabilities_nb(model1, vectorizer1, testSet['text'])
    _, probs2_test = predict_with_probabilities_nb(model2, vectorizer2, testSet['text'])
    
    ensemble_labels_before = ensemble_soft_voting(probs1_test, probs2_test, model1.classes_)
    accuracy_ensemble_before = accuracy_score(testLabels['category'], ensemble_labels_before)

    # Part 3: Retrain models with relabeled Training Set 3
    print("\nRetrained Naive Bayes Model 1 Accuracy")
    combinedSet1 = pd.concat([trainingSet1, trainingSet3])
    model1_retrained, vectorizer1_retrained, accuracy1_after, _ = naiveBayesIsrael(combinedSet1, testSet, testLabels['category'])

    print("\nRetrained Naive Bayes Model 2 Accuracy")
    combinedSet2 = pd.concat([trainingSet2, trainingSet3])
    model2_retrained, vectorizer2_retrained, accuracy2_after, _ = naiveBayesIsrael(combinedSet2, testSet, testLabels['category'])

    # Part 4: Evaluate ensemble model after retraining
    _, probs1_retrain_test = predict_with_probabilities_nb(model1_retrained, vectorizer1_retrained, testSet['text'])
    _, probs2_retrain_test = predict_with_probabilities_nb(model2_retrained, vectorizer2_retrained, testSet['text'])
    
    ensemble_labels_after = ensemble_soft_voting(probs1_retrain_test, probs2_retrain_test, model1_retrained.classes_)
    accuracy_ensemble_after = accuracy_score(testLabels['category'], ensemble_labels_after)

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
def mutualNeuralNetwork(trainingSet1, trainingSet2, trainingSet3, testSet, testLabels):
    
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

    print(f'\nRetrained Neural Network 1 Accuracy: {accuracy1 * 100:.2f}%')
    print(f'\nRetrained Neural Network 2 Accuracy: {accuracy2 * 100:.2f}%')

    report1 = classification_report(testLabels, predictions1, target_names=['0 (Business)', '1 (Entertainment)', '2 (Politics)', '3 (Sport)', '4 (Tech)'])
    report2 = classification_report(testLabels, predictions2, target_names=['0 (Business)', '1 (Entertainment)', '2 (Politics)', '3 (Sport)', '4 (Tech)'])

    print(f'\nRetrained Neural Network 1 Classification Report:\n{report1}')
    print(f'\nRetrained Neural Network 2 Classification Report:\n{report2}')
   

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


def mutualSVM(trainingSet1, trainingSet2, trainingSet3RemovedLabels, testSet, testLabels):
    vectorizer = CountVectorizer()

    # Transform training sets using the single vectorizer instance
    trainingTextVector1 = vectorizer.fit_transform(trainingSet1['text'])
    trainingTextVector2 = vectorizer.transform(trainingSet2['text'])
    trainingTextVector3 = vectorizer.transform(trainingSet3RemovedLabels['text'])
    testTextVector = vectorizer.transform(testSet['text'])

    # Part 1: Train Linear SVM on Training Set 1
    linearSVM = SVC(kernel='linear', probability=True)
    linearSVM.fit(trainingTextVector1, trainingSet1['category'])
    linear_predictions = linearSVM.predict(testTextVector)
    linear_accuracy = accuracy_score(testLabels, linear_predictions)
    linear_report = classification_report(testLabels, linear_predictions,
                                          target_names=['0 (Business)', '1 (Entertainment)', '2 (Politics)',
                                                        '3 (Sport)', '4 (Tech)'])

    # Part 1: Train Non-Linear SVM (Sigmoid Kernel) on Training Set 2
    nonLinearSVM = SVC(kernel='sigmoid', probability=True)
    nonLinearSVM.fit(trainingTextVector2, trainingSet2['category'])
    non_linear_predictions = nonLinearSVM.predict(testTextVector)
    non_linear_accuracy = accuracy_score(testLabels, non_linear_predictions)
    non_linear_report = classification_report(testLabels, non_linear_predictions,
                                              target_names=['0 (Business)', '1 (Entertainment)', '2 (Politics)',
                                                            '3 (Sport)', '4 (Tech)'])

    # Display initial results for Part 1
    print(f"\nLinear SVM Accuracy with Training Set 1: {linear_accuracy * 100:.2f}%")
    print(f"\nLinear SVM Classification Report: \n{linear_report}")

    print(f"\nNon-linear SVM Accuracy with Training Set 2: {non_linear_accuracy * 100:.2f}%")
    print(f"\nNon-Linear SVM Classification Report: \n{non_linear_report}")

    # Calibrate probabilities for mutual learning
    linearSVM_calibrated = CalibratedClassifierCV(linearSVM).fit(trainingTextVector1, trainingSet1['category'])
    nonLinearSVM_calibrated = CalibratedClassifierCV(nonLinearSVM).fit(trainingTextVector2, trainingSet2['category'])

    # Predictions and probabilities for Training Set 3
    linear_predictions3, linear_probabilities = linearSVM_calibrated.predict(
        trainingTextVector3), linearSVM_calibrated.predict_proba(trainingTextVector3)
    non_linear_predictions3, non_linear_probabilities = nonLinearSVM_calibrated.predict(
        trainingTextVector3), nonLinearSVM_calibrated.predict_proba(trainingTextVector3)

    # Relabel Training Set 3
    new_labels = []
    for i in range(len(trainingSet3RemovedLabels)):
        if max(linear_probabilities[i]) > max(non_linear_probabilities[i]):
            new_labels.append(linear_predictions3[i])
        else:
            new_labels.append(non_linear_predictions3[i])

    trainingSet3Relabeled = trainingSet3RemovedLabels.copy()
    trainingSet3Relabeled['category'] = new_labels

    # Combine Training Set 1 with relabeled Training Set 3 for linear SVM retraining
    combinedTrainingSet1 = pd.concat([trainingSet1, trainingSet3Relabeled])
    combinedTextVector1 = vectorizer.transform(combinedTrainingSet1['text'])

    # Combine Training Set 2 with relabeled Training Set 3 for non-linear SVM retraining
    combinedTrainingSet2 = pd.concat([trainingSet2, trainingSet3Relabeled])
    combinedTextVector2 = vectorizer.transform(combinedTrainingSet2['text'])

    # Retrain both SVM models on their respective combined datasets
    linearSVM.fit(combinedTextVector1, combinedTrainingSet1['category'])
    nonLinearSVM.fit(combinedTextVector2, combinedTrainingSet2['category'])

    # Evaluate the retrained SVM models on the test set
    retrained_linear_predictions = linearSVM.predict(testTextVector)
    retrained_linear_accuracy = accuracy_score(testLabels, retrained_linear_predictions)
    retrained_linear_report = classification_report(testLabels, retrained_linear_predictions,
                                                    target_names=['0 (Business)', '1 (Entertainment)', '2 (Politics)',
                                                                  '3 (Sport)', '4 (Tech)'])

    retrained_non_linear_predictions = nonLinearSVM.predict(testTextVector)
    retrained_non_linear_accuracy = accuracy_score(testLabels, retrained_non_linear_predictions)
    retrained_non_linear_report = classification_report(testLabels, retrained_non_linear_predictions,
                                                        target_names=['0 (Business)', '1 (Entertainment)',
                                                                      '2 (Politics)', '3 (Sport)', '4 (Tech)'])

    # Display results for Part 2
    print(f"\nRetrained Linear SVM Accuracy: {retrained_linear_accuracy * 100:.2f}%")
    print(f"\nRetrained Linear SVM Classifcation Report: \n{retrained_linear_report}")

    print(f"\nRetrained Non-linear SVM Accuracy: {retrained_non_linear_accuracy * 100:.2f}%")
    print(f"\nRetrained Non-Linear SVM Classifcation Report: \n{retrained_non_linear_report}")

    return

def mutualNetworkAndSVM(trainingSet1, trainingSet2, trainingSet3, testSet, testLabels):
    # Initial Training and Evaluation
    vectorizerNN = CountVectorizer()
    trainingTextVectorNN = vectorizerNN.fit_transform(trainingSet1['text'])
    testTextVectorNN = vectorizerNN.transform(testSet['text'])
    
    mlpClassifier = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=500, random_state=1)
    mlpClassifier.fit(trainingTextVectorNN, trainingSet1['category'])
    nnInitialPredictions = mlpClassifier.predict(testTextVectorNN)
    nnInitialAccuracy = accuracy_score(testLabels, nnInitialPredictions)
    nnReport = classification_report(testLabels, nnInitialPredictions,
                                              target_names=['0 (Business)', '1 (Entertainment)', '2 (Politics)',
                                                            '3 (Sport)', '4 (Tech)'])
    print(f"\nMLP Neural Network Accuracy on Training Set 1: {nnInitialAccuracy * 100:.2f}%")
    print(f"\nMLP Neural Network Classification Report: \n{nnReport}")


    vectorizerSVM = CountVectorizer()
    trainingTextVectorSVM = vectorizerSVM.fit_transform(trainingSet2['text'])
    testTextVectorSVM = vectorizerSVM.transform(testSet['text'])
    
    svmClassifier = SVC(kernel='sigmoid', probability=True)
    svmClassifier.fit(trainingTextVectorSVM, trainingSet2['category'])
    svmInitialPredictions = svmClassifier.predict(testTextVectorSVM)
    svmInitialAccuracy = accuracy_score(testLabels, svmInitialPredictions)
    svmReport = classification_report(testLabels, svmInitialPredictions,
                                              target_names=['0 (Business)', '1 (Entertainment)', '2 (Politics)',
                                                            '3 (Sport)', '4 (Tech)'])
    print(f"\nNon-Linear SVM Accuracy with Training Set 2: {svmInitialAccuracy * 100:.2f}%")
    print(f"\nNon-Linear SVM Classification Report: \n{svmReport}")

    # Mutual Learning on Training Set 3
    trainingTextVector3NN = vectorizerNN.transform(trainingSet3['text'])
    trainingTextVector3SVM = vectorizerSVM.transform(trainingSet3['text'])
    nnProbabilities3 = mlpClassifier.predict_proba(trainingTextVector3NN)
    svmProbabilities3 = svmClassifier.predict_proba(trainingTextVector3SVM)

    # Relabel based on highest confidence
    mutualLabels = [
        mlpClassifier.predict(trainingTextVector3NN[i])[0] if max(nnProbabilities3[i]) > max(svmProbabilities3[i])
        else svmClassifier.predict(trainingTextVector3SVM[i])[0]
        for i in range(len(trainingSet3))
    ]
    trainingSet3Relabeled = trainingSet3.copy()
    trainingSet3Relabeled['category'] = mutualLabels

    # Retraining Neural Network with Training Set 1 + Relabeled Training Set 3
    combinedTrainingSet1 = pd.concat([trainingSet1, trainingSet3Relabeled])
    combinedTextVectorNN = vectorizerNN.transform(combinedTrainingSet1['text'])
    mlpClassifier.fit(combinedTextVectorNN, combinedTrainingSet1['category'])
    nnRetrainedPredictions = mlpClassifier.predict(testTextVectorNN)
    nnRetrainedAccuracy = accuracy_score(testLabels, nnRetrainedPredictions)
    nnRetrainedReport = classification_report(testLabels, nnRetrainedPredictions,
                                              target_names=['0 (Business)', '1 (Entertainment)', '2 (Politics)', '3 (Sport)', '4 (Tech)'])
    print(f"\nRetrained Neural Network Accuracy: {nnRetrainedAccuracy * 100:.2f}%")
    print(f"\nNeural Network Retrained Classification Report:\n{nnRetrainedReport}")

    # Retraining Non-linear SVM with Training Set 2 + Relabeled Training Set 3
    combinedTrainingSet2 = pd.concat([trainingSet2, trainingSet3Relabeled])
    combinedTextVectorSVM = vectorizerSVM.transform(combinedTrainingSet2['text'])
    svmClassifier.fit(combinedTextVectorSVM, combinedTrainingSet2['category'])
    svmRetrainedPredictions = svmClassifier.predict(testTextVectorSVM)
    svmRetrainedAccuracy = accuracy_score(testLabels, svmRetrainedPredictions)
    svmRetrainedReport = classification_report(testLabels, svmRetrainedPredictions,
                                               target_names=['0 (Business)', '1 (Entertainment)', '2 (Politics)', '3 (Sport)', '4 (Tech)'])
    print(f"\nRetrained Non-linear SVM Accuracy: {svmRetrainedAccuracy * 100:.2f}%")
    print(f"\nNon-linear SVM Retrained Classification Report:\n{svmRetrainedReport}")

#avinash
# Main function to perform mutual learning with Naive Bayes and Linear SVM (with SVM trained on Training Set 2)
def mutualBayesAndSVM(fullTrainingSet, testSet, testLabels):
    # Initialize CountVectorizer
    vectorizer = CountVectorizer()

    # Split the full training set into 3 parts
    trainingSet1, trainingSet2, trainingSet3 = np.array_split(fullTrainingSet, 3)
    trainingSet3RemovedLabels = trainingSet3.drop(columns=['category'])  # Remove labels from Training Set 3

    # Transform training sets using the single vectorizer instance
    trainingTextVector1 = vectorizer.fit_transform(trainingSet1['text'])
    trainingTextVector2 = vectorizer.transform(trainingSet2['text'])
    trainingTextVector3 = vectorizer.transform(trainingSet3RemovedLabels['text'])
    testTextVector = vectorizer.transform(testSet['text'])

    # Part 1: Train Naive Bayes on Training Set 1 and Linear SVM on Training Set 2
    # Train Naive Bayes
    naiveBayesModel = MultinomialNB()
    naiveBayesModel.fit(trainingTextVector1, trainingSet1['category'])
    nb_predictions = naiveBayesModel.predict(testTextVector)
    nb_accuracy = accuracy_score(testLabels, nb_predictions)
    nb_report = classification_report(testLabels, nb_predictions,
                                      target_names=['0 (Business)', '1 (Entertainment)', '2 (Politics)', '3 (Sport)', '4 (Tech)'])

    # Train Linear SVM on Training Set 2
    linearSVM = SVC(kernel='linear', probability=True)
    linearSVM.fit(trainingTextVector2, trainingSet2['category'])
    linear_predictions = linearSVM.predict(testTextVector)
    linear_accuracy = accuracy_score(testLabels, linear_predictions)
    linear_report = classification_report(testLabels, linear_predictions,
                                          target_names=['0 (Business)', '1 (Entertainment)', '2 (Politics)', '3 (Sport)', '4 (Tech)'])

    # Display results for Part 1
    print("\nInitial Evaluation on Test Set:")
    print(f"Naive Bayes Accuracy: {nb_accuracy * 100:.2f}%\n{nb_report}")
    print(f"Linear SVM Accuracy: {linear_accuracy * 100:.2f}%\n{linear_report}")

    # Part 2: Mutual Learning
    # Calibrated probabilities for mutual learning
    nb_calibrated = CalibratedClassifierCV(naiveBayesModel).fit(trainingTextVector1, trainingSet1['category'])
    linearSVM_calibrated = CalibratedClassifierCV(linearSVM).fit(trainingTextVector2, trainingSet2['category'])

    # Predictions and probabilities for Training Set 3
    nb_predictions3, nb_probabilities = nb_calibrated.predict(trainingTextVector3), nb_calibrated.predict_proba(trainingTextVector3)
    linear_predictions3, linear_probabilities = linearSVM_calibrated.predict(trainingTextVector3), linearSVM_calibrated.predict_proba(trainingTextVector3)

    # Relabel Training Set 3 based on highest confidence
    new_labels = []
    for i in range(len(trainingSet3RemovedLabels)):
        if max(nb_probabilities[i]) > max(linear_probabilities[i]):
            new_labels.append(nb_predictions3[i])
        else:
            new_labels.append(linear_predictions3[i])

    trainingSet3Relabeled = trainingSet3RemovedLabels.copy()
    trainingSet3Relabeled['category'] = new_labels

    # Combine Training Set 1 with relabeled Training Set 3
    combinedTrainingSet = pd.concat([trainingSet1, trainingSet3Relabeled])
    combinedTextVector = vectorizer.transform(combinedTrainingSet['text'])  # Reuse vectorizer for consistency

    # Retrain both models on the combined dataset
    naiveBayesModel.fit(combinedTextVector, combinedTrainingSet['category'])
    linearSVM.fit(combinedTextVector, combinedTrainingSet['category'])

    # Evaluate retrained models on the test set
    retrained_nb_predictions = naiveBayesModel.predict(testTextVector)
    retrained_nb_accuracy = accuracy_score(testLabels, retrained_nb_predictions)
    retrained_nb_report = classification_report(testLabels, retrained_nb_predictions,
                                                target_names=['0 (Business)', '1 (Entertainment)', '2 (Politics)', '3 (Sport)', '4 (Tech)'])

    retrained_linear_predictions = linearSVM.predict(testTextVector)
    retrained_linear_accuracy = accuracy_score(testLabels, retrained_linear_predictions)
    retrained_linear_report = classification_report(testLabels, retrained_linear_predictions,
                                                    target_names=['0 (Business)', '1 (Entertainment)', '2 (Politics)', '3 (Sport)', '4 (Tech)'])

    # Display results for Part 2
    print("\nMutual Learning and Retrained Evaluation on Test Set:")
    print(f"Retrained Naive Bayes Accuracy: {retrained_nb_accuracy * 100:.2f}%\n{retrained_nb_report}")
    print(f"Retrained Linear SVM Accuracy: {retrained_linear_accuracy * 100:.2f}%\n{retrained_linear_report}")

    # Summary Table
    print("\nSummary Table of Results:")
    print(f"{'Model':<30} {'Initial Accuracy':<20} {'Retrained Accuracy':<20}")
    print(f"{'Naive Bayes':<30} {nb_accuracy * 100:.2f}%{'':<15} {retrained_nb_accuracy * 100:.2f}%")
    print(f"{'Linear SVM':<30} {linear_accuracy * 100:.2f}%{'':<15} {retrained_linear_accuracy * 100:.2f}%")

#israel
def mutualNetworkAndBayes(trainingSet1, trainingSet2, trainingSet3, testSet, testLabels):
    # Initialize TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

    # Transform training and test sets
    X_train1 = vectorizer.fit_transform(trainingSet1['text'])
    X_train2 = vectorizer.transform(trainingSet2['text'])
    X_train3 = vectorizer.transform(trainingSet3['text'])
    X_test = vectorizer.transform(testSet['text'])
    
    y_train1 = trainingSet1['category']
    y_train2 = trainingSet2['category']
    y_test = testLabels['category']
    
    # Initialize models
    nb_model = MultinomialNB()
    mlp_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=10, warm_start=True, random_state=42)
    
    # Check for existing weights
    if os.path.exists('best_nb_model.pkl') and os.path.exists('best_mlp_model.pkl'):
        nb_model = joblib.load('best_nb_model.pkl')
        mlp_model = joblib.load('best_mlp_model.pkl')
        print("Loaded previous weights for both models.")
    else:
        print("Training models from scratch.")

    # Track best accuracy
    best_val_accuracy = 0.0
    patience, patience_counter = 3, 0
    
    # Initial performance
    print("\nInitial Naive Bayes Accuracy:")
    nb_model.fit(X_train1, y_train1)
    nb_initial_accuracy = accuracy_score(y_test, nb_model.predict(X_test))
    print(f"Naive Bayes Initial Accuracy: {nb_initial_accuracy * 100:.2f}%")
    
    print("\nInitial MLP Accuracy:")
    mlp_model.fit(X_train2, y_train2)
    mlp_initial_accuracy = accuracy_score(y_test, mlp_model.predict(X_test))
    print(f"MLP Initial Accuracy: {mlp_initial_accuracy * 100:.2f}%")
    
    # Mutual Learning
    for epoch in range(10):
        print(f"\n--- Epoch {epoch + 1}/10 ---")
        nb_predictions = nb_model.predict(X_train3)
        mlp_predictions = mlp_model.predict(X_train3)
        
        # Generate pseudo-labels for Training Set 3
        pseudo_labels = [
            nb_predictions[i] if max(nb_model.predict_proba(X_train3)[i]) > max(mlp_model.predict_proba(X_train3)[i])
            else mlp_predictions[i]
            for i in range(len(trainingSet3))
        ]
        
        # Retrain models with Training Set 3 and pseudo-labels
        combined_X_train1 = vectorizer.transform(pd.concat([trainingSet1, trainingSet3])['text'])
        combined_y_train1 = pd.concat([y_train1, pd.Series(pseudo_labels)])
        nb_model.partial_fit(combined_X_train1, combined_y_train1)
        
        combined_X_train2 = vectorizer.transform(pd.concat([trainingSet2, trainingSet3])['text'])
        combined_y_train2 = pd.concat([y_train2, pd.Series(pseudo_labels)])
        mlp_model.partial_fit(combined_X_train2, combined_y_train2)
        
        # Evaluate on test set
        val_predictions = mlp_model.predict(X_test)
        val_accuracy = accuracy_score(y_test, val_predictions)
        print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
        
        # Save best models
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            joblib.dump(nb_model, 'best_nb_model.pkl')
            joblib.dump(mlp_model, 'best_mlp_model.pkl')
            print(f"New checkpoint saved at epoch {epoch + 1}.")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping due to no improvement.")
                break
    
    # Load best models and evaluate final performance
    nb_model = joblib.load('best_nb_model.pkl')
    mlp_model = joblib.load('best_mlp_model.pkl')
    
    nb_final_accuracy = accuracy_score(y_test, nb_model.predict(X_test))
    mlp_final_accuracy = accuracy_score(y_test, mlp_model.predict(X_test))
    
    print("\nFinal Naive Bayes Accuracy:")
    print(f"Naive Bayes Final Accuracy: {nb_final_accuracy * 100:.2f}%")
    print("\nFinal MLP Accuracy:")
    print(f"MLP Final Accuracy: {mlp_final_accuracy * 100:.2f}%")
    
    # Percentage improvement
    nb_improvement = ((nb_final_accuracy - nb_initial_accuracy) / nb_initial_accuracy) * 100
    mlp_improvement = ((mlp_final_accuracy - mlp_initial_accuracy) / mlp_initial_accuracy) * 100
    print(f"\nNaive Bayes Accuracy Improvement: {nb_improvement:.2f}%")
    print(f"MLP Accuracy Improvement: {mlp_improvement:.2f}%")


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
    trainingSet3RemovedLabels = trainingSet3.drop(columns=['category'])

    # Save the preprocessed data to new CSV files.
    fullTrainingSet.to_csv('BBC_train_full_preprocessed.csv', index=False)
    testSet.to_csv('test_data_preprocessed.csv', index=False)
    print('\nSuccessfully preprocessed the data.\n')

    # Full Naive Bayes Model
    #naiveBayes(fullTrainingSet, testSet, testLabels['category'])

    # Full MLP Neural Network Model
    #neuralNetwork(fullTrainingSet, testSet, testLabels['category'])

    # Full SVM models
    #trainSVM(fullTrainingSet, testSet, testLabels['category'], kernel_type='linear')
    #trainSVM(fullTrainingSet, testSet, testLabels['category'], kernel_type='sigmoid')
    
    # Homogenous Neural Network Model
    #mutualNeuralNetwork(trainingSet1, trainingSet2, trainingSet3RemovedLabels, testSet, testLabels['category'])
    
    # Homogenous Naive Bayes Model
    #mutualNaiveBayes(trainingSet1, trainingSet2, trainingSet3, testSet, testLabels)

    # Homogenous SVM Model
    #mutualSVM(trainingSet1, trainingSet2, trainingSet3RemovedLabels, testSet, testLabels['category'])

    # Mutual Learning between MLP Neural Network and Non-Linear (Sigmoid) SVM 
    #mutualNetworkAndSVM(trainingSet1, trainingSet2, trainingSet3, testSet, testLabels)

    # Mutual Learning between Naive Bayes and Linear (Sigmoid) SVM
    #mutualBayesAndSVM(fullTrainingSet, testSet, testLabels['category'])

    # Mutual Learning between MLP Neural Network and Naive Bayes
    mutualNetworkAndBayes(trainingSet1, trainingSet2, trainingSet3, testSet, testLabels) 

if __name__ == '__main__':
    main()

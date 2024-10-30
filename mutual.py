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

if __name__ == '__main__':
    main()

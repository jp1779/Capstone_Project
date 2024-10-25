import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
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
  
  
  
  
  
  
  
      # Naive Bayes Model function
def naiveBayes(trainingSet, testSet, testLabels, use_tfidf=False):
    # Obtain the data
    trainingText = trainingSet['text']
    trainingCategory = trainingSet['category']
    testText = testSet['text']

    # Vectorize the text
    if use_tfidf:
        vectorizer = TfidfVectorizer()
    else:
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
    
    # Homogenous Neural Network Model
    mutualNeuralNetwork(trainingSet1, trainingSet2, trainingSet3RemovedLabels, testSet, testLabels['category'])

    # Homogenous Naive Bayes Model
    

    
    
    
    # Part 1: Train Naive Bayes models on trainingSet1 and trainingSet2
    print("\nTraining Naive Bayes model on Training Set 1")
    model1, vectorizer1, accuracy1, report1 = naiveBayes(trainingSet1, testSet, testLabels['category'], use_tfidf=True)

    print("\nTraining Naive Bayes model on Training Set 2")
    model2, vectorizer2, accuracy2, report2 = naiveBayes(trainingSet2, testSet, testLabels['category'], use_tfidf=True)

    # Use both models to predict labels and probabilities for trainingSet3
    labels1, probs1 = predict_with_probabilities_nb(model1, vectorizer1, trainingSet3NoLabels['text'])
    labels2, probs2 = predict_with_probabilities_nb(model2, vectorizer2, trainingSet3NoLabels['text'])

    # Set a lower confidence threshold
    threshold = 0.8
    new_labels = []
    indices_to_include = []

    # Include data points where either model predicts with high confidence
    for i in range(len(trainingSet3NoLabels)):
        max_prob1 = max(probs1[i])
        max_prob2 = max(probs2[i])

        if max_prob1 >= threshold:
            new_labels.append(labels1[i])
            indices_to_include.append(i)
        elif max_prob2 >= threshold:
            new_labels.append(labels2[i])
            indices_to_include.append(i)
        else:
            # Skip low-confidence predictions
            pass

    # Filter trainingSet3 to only include high-confidence predictions
    trainingSet3Filtered = trainingSet3NoLabels.iloc[indices_to_include].copy()
    trainingSet3Filtered['category'] = new_labels

    # Print the number of data points added from Training Set 3
    print(f"\nNumber of high-confidence labels added from Training Set 3: {len(new_labels)}")

    # Combine Training Set 1 and the filtered Training Set 3
    combinedSet1 = pd.concat([trainingSet1, trainingSet3Filtered])

    # Retrain the Naive Bayes model with the adjusted combined set
    print("\nRetraining Naive Bayes model 1 with Training Set 1 and high-confidence labels from Training Set 3")

    # Use TF-IDF vectorizer fitted on the combined training data
    vectorizer_combined1 = TfidfVectorizer()
    combinedTextVector1 = vectorizer_combined1.fit_transform(combinedSet1['text'])
    testTextVector1 = vectorizer_combined1.transform(testSet['text'])

    # Retrain the model
    naiveClassifier1_retrained = MultinomialNB()
    naiveClassifier1_retrained.fit(combinedTextVector1, combinedSet1['category'])

    # Evaluate the retrained model
    categoryPredictions1_retrained = naiveClassifier1_retrained.predict(testTextVector1)
    accuracy1_retrained = accuracy_score(testLabels['category'], categoryPredictions1_retrained)
    report1_retrained = classification_report(testLabels['category'], categoryPredictions1_retrained,
                                              target_names=['0 (Business)', '1 (Entertainment)',
                                                            '2 (Politics)', '3 (Sport)', '4 (Tech)'])

    print(f"\nRetrained Naive Bayes Model 1 Accuracy: {accuracy1_retrained * 100:.2f}%")
    print(f'\nRetrained Naive Bayes Model 1 Classification Report:\n{report1_retrained}')

    # Repeat the process for Model 2
    combinedSet2 = pd.concat([trainingSet2, trainingSet3Filtered])

    print("\nRetraining Naive Bayes model 2 with Training Set 2 and high-confidence labels from Training Set 3")

    # Use TF-IDF vectorizer fitted on the combined training data
    vectorizer_combined2 = TfidfVectorizer()
    combinedTextVector2 = vectorizer_combined2.fit_transform(combinedSet2['text'])
    testTextVector2 = vectorizer_combined2.transform(testSet['text'])

    # Retrain the model
    naiveClassifier2_retrained = MultinomialNB()
    naiveClassifier2_retrained.fit(combinedTextVector2, combinedSet2['category'])

    # Evaluate the retrained model
    categoryPredictions2_retrained = naiveClassifier2_retrained.predict(testTextVector2)
    accuracy2_retrained = accuracy_score(testLabels['category'], categoryPredictions2_retrained)
    report2_retrained = classification_report(testLabels['category'], categoryPredictions2_retrained,
                                              target_names=['0 (Business)', '1 (Entertainment)',
                                                            '2 (Politics)', '3 (Sport)', '4 (Tech)'])

    print(f"\nRetrained Naive Bayes Model 2 Accuracy: {accuracy2_retrained * 100:.2f}%")
    print(f'\nRetrained Naive Bayes Model 2 Classification Report:\n{report2_retrained}')

    # Report the initial and retrained accuracies
    print(f"\nInitial Naive Bayes Model 1 Accuracy: {accuracy1 * 100:.2f}%")
    print(f"Retrained Naive Bayes Model 1 Accuracy: {accuracy1_retrained * 100:.2f}%")

    print(f"\nInitial Naive Bayes Model 2 Accuracy: {accuracy2 * 100:.2f}%")
    print(f"Retrained Naive Bayes Model 2 Accuracy: {accuracy2_retrained * 100:.2f}%")
    
    

if __name__ == '__main__':
    main()

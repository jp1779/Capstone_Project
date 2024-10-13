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

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Initialize the stemmer
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

# Function that will preprocess the data so that the ML models can use it.
# Removes numbers, removes symbols, makes all text lowercase,
# tokenizes the text, removes stopwords, and applies lemmatization.
def PreprocessText(text):

    # Load stopwords from nltk
    stop_words = set(stopwords.words('english'))

    # Remove numbers and symbols from the data.
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove consecutive blank spaces between words and trailing/leading whitespace.
    text = re.sub(r'\s+', ' ', text).strip()

    # Makes the text lowercase.
    text = text.lower()

    # Tokenize the data using nltk's word_tokenize.
    tokenizedData = word_tokenize(text)

    # Remove stopwords from the tokenized data.
    tokenizedData = [word for word in tokenizedData if word not in stop_words]

    # Get POS tags for each token
    pos_tags = nltk.pos_tag(tokenizedData)

    # Apply stemming
    tokenizedData = [stemmer.stem(word, get_wordnet_pos(tag)) for word, tag in pos_tags]

    # Apply lemmatization with the correct POS tag
    tokenizedData = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]

    return ' '.join(lemmatized_tokens)  # Return the lemmatized data as a list

# Naive Bayes Model
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

# Naive Bayes Model
def naiveBayes(trainingSet, testSet, testLabels):
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

    print(f'\nNaive Bayes Accuracy: {accuracy * 100:.2f}%')
    print(f'\nNaive Bayes Classification Report:\n{report}')
    return accuracy

# MLP Neural Network Model
def neuralNetwork(trainingSet, testSet, testLabels):
    trainingText = trainingSet['text']
    trainingCategory = trainingSet['category']
    testText = testSet['text']

    # Vectorize the text
    vectorizer = CountVectorizer()
    trainingTextVector = vectorizer.fit_transform(trainingText)
    testTextVector = vectorizer.transform(testText)

    # Train the MLP Neural Network
    mlpModel = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=500, random_state=1)
    mlpModel.fit(trainingTextVector, trainingCategory)

    # Predict and evaluate
    categoryPredictions = mlpModel.predict(testTextVector)
    accuracy = accuracy_score(testLabels, categoryPredictions)
    report = classification_report(testLabels, categoryPredictions, target_names=['0 (Business)', '1 (Entertainment)', '2 (Politics)', '3 (Sport)', '4 (Tech)'])

    print(f'\nMLP Neural Network Accuracy: {accuracy * 100:.2f}%')
    print(f'\nMLP Neural Network Classification Report:\n{report}')
    return accuracy

# SVM Model
def trainSVM(trainingSet, testSet, testLabels, kernel_type):
    trainingText = trainingSet['text']
    trainingCategory = trainingSet['category']
    testText = testSet['text']

    # Vectorize the text
    vectorizer = CountVectorizer()
    trainingTextVector = vectorizer.fit_transform(trainingText)
    testTextVector = vectorizer.transform(testText)

    # Train the SVM model
    svmModel = SVC(kernel=kernel_type, probability=True)
    svmModel.fit(trainingTextVector, trainingCategory)

    # Predict and evaluate
    categoryPredictions = svmModel.predict(testTextVector)
    accuracy = accuracy_score(testLabels, categoryPredictions)
    report = classification_report(testLabels, categoryPredictions, target_names=['0 (Business)', '1 (Entertainment)', '2 (Politics)', '3 (Sport)', '4 (Tech)'])

    print(f'\nSVM ({kernel_type}) Accuracy: {accuracy * 100:.2f}%')
    print(f'\nSVM ({kernel_type}) Classification Report:\n{report}')
    return accuracy

# Function to relabel Training Set 3 based on posterior probabilities
def relabel_training_set(train1, train2, train3_no_labels, vectorizer):
    train1_vector = vectorizer.transform(train1['text'])
    train2_vector = vectorizer.transform(train2['text'])

    nb_model_1 = MultinomialNB()
    nb_model_1.fit(train1_vector, train1['category'])

    nb_model_2 = MultinomialNB()
    nb_model_2.fit(train2_vector, train2['category'])

    train3_vector = vectorizer.transform(train3_no_labels['text'])
    probs_model_1 = nb_model_1.predict_proba(train3_vector)
    probs_model_2 = nb_model_2.predict_proba(train3_vector)

    new_labels = []
    for i in range(len(train3_no_labels)):
        if max(probs_model_1[i]) > max(probs_model_2[i]):
            new_labels.append(nb_model_1.predict(train3_vector[i])[0])
        else:
            new_labels.append(nb_model_2.predict(train3_vector[i])[0])

    train3_no_labels['category'] = new_labels
    return train3_no_labels

# Function to retrain and evaluate models
def retrain_and_evaluate(train1, relabeled_train3, test_set, test_labels, vectorizer):
    combined_train = pd.concat([train1, relabeled_train3])

    combined_vectors = vectorizer.fit_transform(combined_train['text'])
    test_vectors = vectorizer.transform(test_set['text'])

    final_nb_model = MultinomialNB()
    final_nb_model.fit(combined_vectors, combined_train['category'])
    final_predictions = final_nb_model.predict(test_vectors)

    final_accuracy = accuracy_score(test_labels, final_predictions)
    final_report = classification_report(test_labels, final_predictions)

    print(f"Final Naive Bayes Accuracy after mutual learning: {final_accuracy * 100:.2f}%")
    print(f"\nFinal Naive Bayes Classification Report:\n{final_report}")

# Main function with mutual learning toggle
def main(mutual_learning_enabled=False):
    train_data = pd.read_csv('BBC_train_full.csv')
    test_data = pd.read_csv('test_data.csv')
    test_labels = test_data['category']

    print("Preprocessing data...")
    train_data['text'] = train_data['text'].apply(preprocess_text)
    test_data['text'] = test_data['text'].apply(preprocess_text)

    train1, train2, train3 = np.array_split(train_data, 3)
    train3_no_labels = train3.drop(columns=['category'])

    print("Training Naive Bayes models...")
    accuracy1, _, vectorizer1 = train_and_evaluate(train1, test_data, test_labels)
    accuracy2, _, vectorizer2 = train_and_evaluate(train2, test_data, test_labels)

    if mutual_learning_enabled:
        print("Mutual learning enabled. Relabeling Training Set 3...")
        relabeled_train3 = relabel_training_set(train1, train2, train3_no_labels, vectorizer1)
        retrain_and_evaluate(train1, relabeled_train3, test_data, test_labels, vectorizer1)
    else:
        print("Mutual learning disabled. Running standard Naive Bayes on the full training set...")
        combined_train = pd.concat([train1, train2, train3])
        train_and_evaluate(combined_train, test_data, test_labels)

    print("Running MLP Neural Network...")
    neuralNetwork(train_data, test_data, test_labels)

    print("Running SVM (Linear) model...")
    trainSVM(train_data, test_data, test_labels, kernel_type='linear')

 

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
    main(mutual_learning_enabled=True)

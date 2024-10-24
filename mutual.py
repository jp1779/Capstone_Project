import re
import pandas as pd
import numpy as np
import nltk

# Download necessary NLTK data packages
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

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

    return ' '.join(tokenizedData)  # Return the processed data as a string

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
    # Read the provided CSV files
    fullTrainingSet = pd.read_csv('BBC_train_full.csv')
    testSet = pd.read_csv('test_data.csv')
    testLabels = pd.read_csv('test_labels.csv')

    # Apply preprocessing to the training and testing sets
    print("Preprocessing the data...")
    fullTrainingSet['text'] = fullTrainingSet['text'].apply(PreprocessText)
    testSet['text'] = testSet['text'].apply(PreprocessText)

    # Split the full training set into 3 equal-sized subsets
    trainingSet1, trainingSet2, trainingSet3 = np.array_split(fullTrainingSet, 3)

    # Part 1: Train Naive Bayes models on trainingSet1 and trainingSet2
    print("\nTraining Naive Bayes model on Training Set 1")
    model1, vectorizer1, accuracy1, report1 = naiveBayes(trainingSet1, testSet, testLabels['category'], use_tfidf=True)

    print("\nTraining Naive Bayes model on Training Set 2")
    model2, vectorizer2, accuracy2, report2 = naiveBayes(trainingSet2, testSet, testLabels['category'], use_tfidf=True)

    # Part 2: Mutual Learning between the two Naive Bayes models
    # Remove labels from trainingSet3
    trainingSet3NoLabels = trainingSet3.drop(columns=['category'])

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

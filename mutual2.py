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

# Ensemble method to combine model predictions using soft voting
def ensemble_soft_voting(probs1, probs2, classes_):
    avg_probs = (probs1 + probs2) / 2  # Average the probabilities from both models
    ensemble_label_indices = avg_probs.argmax(axis=1)  # Choose the label with the highest average probability
    ensemble_labels = [classes_[i] for i in ensemble_label_indices]  # Convert indices back to class labels
    return ensemble_labels

# Main function with soft voting ensemble method
def main_with_ensemble():
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
    model1, vectorizer1, accuracy1_before, report1 = naiveBayes(trainingSet1, testSet, testLabels['category'])

    print("\nTraining Naive Bayes model on Training Set 2")
    model2, vectorizer2, accuracy2_before, report2 = naiveBayes(trainingSet2, testSet, testLabels['category'])

    # Part 2: Evaluate ensemble model before retraining
    _, probs1_test = predict_with_probabilities_nb(model1, vectorizer1, testSet['text'])
    _, probs2_test = predict_with_probabilities_nb(model2, vectorizer2, testSet['text'])
    
    print("\nEnsemble performance before retraining:")
    ensemble_labels_before = ensemble_soft_voting(probs1_test, probs2_test, model1.classes_)
    accuracy_ensemble_before = accuracy_score(testLabels['category'], ensemble_labels_before)
    print(f'\nEnsemble Accuracy before retraining: {accuracy_ensemble_before * 100:.2f}%')

    # Part 3: Retrain models with relabeled Training Set 3
    print("\nRetraining Naive Bayes model 1 with relabeled Training Set 3")
    combinedSet1 = pd.concat([trainingSet1, trainingSet3])
    model1_retrained, vectorizer1_retrained, accuracy1_after, _ = naiveBayes(combinedSet1, testSet, testLabels['category'])

    print("\nRetraining Naive Bayes model 2 with relabeled Training Set 3")
    combinedSet2 = pd.concat([trainingSet2, trainingSet3])
    model2_retrained, vectorizer2_retrained, accuracy2_after, _ = naiveBayes(combinedSet2, testSet, testLabels['category'])

    # Part 4: Evaluate ensemble model after retraining
    _, probs1_retrain_test = predict_with_probabilities_nb(model1_retrained, vectorizer1_retrained, testSet['text'])
    _, probs2_retrain_test = predict_with_probabilities_nb(model2_retrained, vectorizer2_retrained, testSet['text'])
    
    print("\nEnsemble performance after retraining:")
    ensemble_labels_after = ensemble_soft_voting(probs1_retrain_test, probs2_retrain_test, model1_retrained.classes_)
    accuracy_ensemble_after = accuracy_score(testLabels['category'], ensemble_labels_after)
    print(f'\nEnsemble Accuracy after retraining: {accuracy_ensemble_after * 100:.2f}%')

    # Calculate percentage increases for Model 1, Model 2, and Ensemble
    percentage_increase_model1 = ((accuracy1_after - accuracy1_before) / accuracy1_before) * 100
    percentage_increase_model2 = ((accuracy2_after - accuracy2_before) / accuracy2_before) * 100
    percentage_increase_ensemble = ((accuracy_ensemble_after - accuracy_ensemble_before) / accuracy_ensemble_before) * 100

    print(f"\nModel 1 Accuracy increased by {percentage_increase_model1:.2f}%")
    print(f"Model 2 Accuracy increased by {percentage_increase_model2:.2f}%")
    print(f"Ensemble Accuracy increased by {percentage_increase_ensemble:.2f}%")

if __name__ == '__main__':
    main_with_ensemble()

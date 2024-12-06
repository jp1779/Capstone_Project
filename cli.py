# Import required functions and libraries
import pandas as pd
import numpy as np
from mutual import (
    naiveBayes, neuralNetwork, trainSVM, mutualNaiveBayes, 
    mutualNeuralNetwork, mutualSVM, mutualNetworkAndSVM, 
    mutualBayesAndSVM, mutualNetworkAndBayes, PreprocessText
)

def display_menu():
    print("\nMain Menu")
    print("1. Naive Bayes Full Training")
    print("2. MLP Neural Network Full Training")
    print("3. Linear SVM (Linear Kernel) Full Training")
    print("4. Non-Linear SVM (Sigmoid Kernel) Full Training")
    print("5. Homogenous Mutual Learning: Naive Bayes")
    print("6. Homogenous Mutual Learning: Neural Network")
    print("7. Homogenous Mutual Learning: (Linear/Non-Linear): SVM")
    print("8. Mutual Learning: Neural Network and Non-Linear SVM")
    print("9. Mutual Learning: Naive Bayes and Linear SVM")
    print("10. Mutual Learning: Neural Network and Naive Bayes")
    print("e. Exit")
    print("\nEnter your choice: ", end='')

def cli(fullTrainingSet, testSet, testLabels, trainingSet1, trainingSet2, trainingSet3, trainingSet3RemovedLabels):
    while True:
        display_menu()
        choice = input().strip()
        
        if choice == '1':
            naiveBayes(fullTrainingSet, testSet, testLabels['category'])
        elif choice == '2':
            neuralNetwork(fullTrainingSet, testSet, testLabels['category'])
        elif choice == '3':
            trainSVM(fullTrainingSet, testSet, testLabels['category'], kernel_type='linear')
        elif choice == '4':
            trainSVM(fullTrainingSet, testSet, testLabels['category'], kernel_type='sigmoid')
        elif choice == '5':
            mutualNaiveBayes(trainingSet1, trainingSet2, trainingSet3, testSet, testLabels)
        elif choice == '6':
            mutualNeuralNetwork(trainingSet1, trainingSet2, trainingSet3, testSet, testLabels['category'])
        elif choice == '7':
            mutualSVM(trainingSet1, trainingSet2, trainingSet3RemovedLabels, testSet, testLabels['category'])
        elif choice == '8':
            mutualNetworkAndSVM(trainingSet1, trainingSet2, trainingSet3, testSet, testLabels)
        elif choice == '9':
            mutualBayesAndSVM(fullTrainingSet, testSet, testLabels['category'])
        elif choice == '10':
            mutualNetworkAndBayes(trainingSet1, trainingSet2, trainingSet3, testSet, testLabels)
        elif choice.lower() == 'e':
            print("Exiting")
            break
        else:
            print("Invalid choice. Please enter a number between 1-10 or 'e' to exit.")

if __name__ == '__main__':
    # Preload and preprocess datasets
    fullTrainingSet = pd.read_csv('BBC_train_full.csv')
    testSet = pd.read_csv('test_data.csv')
    testLabels = pd.read_csv('test_labels.csv')

    # Preprocess text data
    fullTrainingSet['text'] = fullTrainingSet['text'].apply(PreprocessText)
    testSet['text'] = testSet['text'].apply(PreprocessText)

    # Convert tokenized text back to a single string
    fullTrainingSet['text'] = fullTrainingSet['text'].apply(lambda tokens: ' '.join(tokens))
    testSet['text'] = testSet['text'].apply(lambda tokens: ' '.join(tokens))

    # Split the full training set into subsets
    trainingSet1, trainingSet2, trainingSet3 = np.array_split(fullTrainingSet, 3)
    trainingSet3RemovedLabels = trainingSet3.drop(columns=['category'])

    print("\nWelcome to the Mutual Learning CLI!")
    print("By: Jorge Puga, Avinash Pandey, Israel Oladeji, Brendan Tea")
    cli(fullTrainingSet, testSet, testLabels, trainingSet1, trainingSet2, trainingSet3, trainingSet3RemovedLabels)

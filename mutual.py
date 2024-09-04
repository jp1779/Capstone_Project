import re # For regular expression operations like removing spaces.
import pandas as pd # For CSV files operations.

import nltk # For extra preprocessing functions
from nltk.tokenize import word_tokenize # Tokenizes the data


# Function called just to see if the other functions work on a random string.
def Test():
  test = 'Thi s is a TeSTING str ing, 5'
  print('\nOriginal string: ' + test)

  preProcessedTest = PreprocessText(test)
  print('Preprocessed: ' , preProcessedTest)
  




# Function that will preprocess the text as required.
def PreprocessText(text):

  # Remove the blank spaces in the text and return text without whitespace.
  # Uses the sub function to replace whitespace character sequences and then use strip to remove leading/trailing whitespaces.
  text = re.sub(r'\s+', ' ', text).strip()

  # Makes the text lowercase.
  text = text.lower()

  # Tokenize the data (new var since it is of type list[str] now).
  tokenizedData = word_tokenize(text)

  return tokenizedData



def main():

  Test()

  # Read the provided CSV files.
  fullTrainingSet = pd.read_csv('BBC_train_full.csv')
  testSet = pd.read_csv('test_data.csv')

  # Now apply to the training and testing sets.
  fullTrainingSet['text'] = fullTrainingSet['text'].apply(PreprocessText)
  testSet['text'] = testSet['text'].apply(PreprocessText)
  
  # Save this as a new file so we can check if it looks good.
  fullTrainingSet.to_csv('BBC_train_full_preprocessed.csv', index = False)
  testSet.to_csv('test_data_preprocessed.csv', index = False)

  print('\nSaved the training and test sets with no spaces\n')



if __name__ == '__main__':
  main()
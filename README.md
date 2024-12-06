# CSCI 49500 Capstone Project

### Group Members <br/>
**Jorge Puga Hernandez**  
**Avinash Pandey**  
**Israel Oladeji**  
**Brendan Tea**
<br/>

### Overview
Machine learning models often require vast resources and large datasets to achieve high accuracy, which isn't 
always feasible when resources are limited. Our project focuses on exploring an alternative approach to address
this challenge which is by using the mutual learning algorithm for news classification.

This algorithm allows multiple machine learning models to collaborate and share information, which in turn improves
their individual classification accuracy. Essentially, the mutual learning approach enables models to learn from each
other, enhancing overall performance without needing extensive data or resources.

The mutual learning algorithm works by training models on different subsets of the data. These models then refine
themselves based on the information shared by the other models in a "teacher-student" environment. This setup allows
us to use multiple smaller datasets instead of relying on one large dataset, making it more efficient when resources are scarce.

By the end of the project, we aim to demonstrate that this mutual learning approach can indeed enhance the performance
of the models we are using, showing its potential as an effective solution in resource-constrained environments.

### Project Structure
```
mutual.py                # Core implementation of all models and mutual learning functions
cli.py                   # Command-line interface for interacting with the program.
BBC_train_full.csv       # Full training dataset
test_data.csv            # Test dataset
test_labels.csv          # Test labels
README.md                # Project documentation
```
The datasets should be in the same directory as the code so that they can be used by the program. The preprocessed versions of the
datasets are included for viewing but are not needed to make the program work, since it will generate them on its own.

### Code
*mutual.py* is where all the functions that will be called are contained. It contains all the data preprocessing techniques,
full training set evaluations, homogenous mutual learning evaluations, and heterogenous mutual learning evaluations.

There are a lot of imported libraries that will be needed for the code to work. These are all listed on mutual.py and can
be installed using the command below in your terminal (or any other method):

```
pip install [library name]
```

Since there are a lot of parts to this project, a command line interface called *cli.py* has been made to make it easier to
interact with the functions in *mutual.py*. To RUN the program, just use the following command (replace python with python3 if needed):

```
python cli.py
```

This will start the command line interface which you will then pick an option from:
```
Main Menu
1. Naive Bayes Full Training
2. MLP Neural Network Full Training
3. Linear SVM (Linear Kernel) Full Training
4. Non-Linear SVM (Sigmoid Kernel) Full Training
5. Homogenous Mutual Learning: Naive Bayes
6. Homogenous Mutual Learning: Neural Network
7. Homogenous Mutual Learning: (Linear/Non-Linear): SVM
8. Mutual Learning: Neural Network and Non-Linear SVM
9. Mutual Learning: Naive Bayes and Linear SVM
10. Mutual Learning: Neural Network and Naive Bayes
e. Exit

Enter your choice:
```

It is worth noting that preprocessing and the training of the models might take a few seconds.
Please be patient if it is the first time running the program!
A pycache folder will be generated in your directory to store compiled Python files for faster execution.
Once an option is chosen, you will see output showing you the evaluation results.

### Project Segments
![processMutualLearning](https://github.com/user-attachments/assets/bf5df7a7-a30c-4ddc-a730-26762934d03c)

### Data Preprocessing
![DataPreprocessing](https://github.com/user-attachments/assets/97601efb-cf18-4c6c-8187-009390c47884)

### Results

Full Training Set Results:

![FullTrainingSetResults](https://github.com/user-attachments/assets/7ec6fe4e-1bff-4768-b73b-632c09179511)

Homogenous Mutual Learning:

![HomogenousMutualLearningResults](https://github.com/user-attachments/assets/e2e26e58-fb66-454b-8d0f-8f802aa7e757)

Naive Bayes and Linear SVM Mutual Learning:

![NB_LinSVM](https://github.com/user-attachments/assets/9dd5e50b-ca6c-4c78-bae4-e7ad383f73c3)

MLP Neural Network and Non-Linear SVM Mutual Learning:

![MLP_NonLinSVM](https://github.com/user-attachments/assets/306a62db-db5e-43ff-b1f2-be9b699ac47d)

MLP Neural Network and Naive Bayes Mutual Learning:

![MLP_NB](https://github.com/user-attachments/assets/faab8d6b-646d-4f82-9894-a0b32a4d66cd)


### Additional Resources
[SabrinaReport.pdf](https://github.com/user-attachments/files/17000707/CapstoneBookShortened.pdf)

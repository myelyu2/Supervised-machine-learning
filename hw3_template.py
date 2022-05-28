#%%
import numpy as np
import pandas as pd
import nltk
import sklearn 
import string
import re # helps you filter urls
from sklearn.metrics import accuracy_score
#%%
#Whether to test your Q9 for not? Depends on correctness of all modules
def test_pipeline():
    return True # Make this true when all tests pass

# Convert part of speech tag from nltk.pos_tag to word net compatible format
# Simple mapping based on first letter of return tag to make grading consistent
# Everything else will be considered noun 'n'
posMapping = {
# "First_Letter by nltk.pos_tag":"POS_for_lemmatizer"
    "N":'n',
    "V":'v',
    "J":'a',
    "R":'r'
}

#%%
def process(text, lemmatizer=nltk.stem.wordnet.WordNetLemmatizer()):
    """ Normalizes case and handles punctuation
    Inputs:
        text: str: raw text
        lemmatizer: an instance of a class implementing the lemmatize() method
                    (the default argument is of type nltk.stem.wordnet.WordNetLemmatizer)
    Outputs:
        list(str): tokenized text
    """
    cleared_text = text

    
    # Clear the string
    cleared_text = cleared_text.lower()
    cleared_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', "", cleared_text)
    cleared_text = cleared_text.replace("'s", "")
    cleared_text = cleared_text.replace("'", "")
    cleared_text = cleared_text.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
    
    tokens = nltk.word_tokenize(cleared_text)
    tags = nltk.pos_tag(tokens)
    
    for i in range(len(tokens)):
        if tags[i][1][0] in posMapping:
            tokens[i] = lemmatizer.lemmatize(tokens[i], posMapping[tags[i][1][0]])
        else:
            try:
                tokens[i] = lemmatizer.lemmatize(tokens[i], 'n')
            except:
                pass
        
    return(tokens)
    
#%%
def process_all(df, lemmatizer=nltk.stem.wordnet.WordNetLemmatizer()):
    """ process all text in the dataframe using process function.
    Inputs
        df: pd.DataFrame: dataframe containing a column 'text' loaded from the CSV file
        lemmatizer: an instance of a class implementing the lemmatize() method
                    (the default argument is of type nltk.stem.wordnet.WordNetLemmatizer)
    Outputs
        pd.DataFrame: dataframe in which the values of text column have been changed from str to list(str),
                        the output from process_text() function. Other columns are unaffected.
    """
    copy_data = df.copy()
    copy_data['text'] = copy_data['text'].apply(process)

    return copy_data
    
#%%
def create_features(processed_tweets, stop_words):
    """ creates the feature matrix using the processed tweet text
    Inputs:
        processed_tweets: pd.DataFrame: processed tweets read from train/test csv file, containing the column 'text'
        stop_words: list(str): stop_words by nltk stopwords (after processing)
    Outputs:
        sklearn.feature_extraction.text.TfidfVectorizer: the TfidfVectorizer object used
            we need this to tranform test tweets in the same way as train tweets
        scipy.sparse.csr.csr_matrix: sparse bag-of-words TF-IDF feature matrix
    """
    def id(text):
        return text
    
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False, tokenizer=id, stop_words=stop_words, min_df=2)
    X = vectorizer.fit_transform(processed_tweets['text']) 
    
    return vectorizer, X

#%%
def create_labels(processed_tweets):
    """ creates the class labels from screen_name
    Inputs:
        processed_tweets: pd.DataFrame: tweets read from train file, containing the column 'screen_name'
    Outputs:
        numpy.ndarray(int): dense binary numpy array of class labels
    """
    labels = processed_tweets[['screen_name']].to_numpy().ravel()
    for i in range(len(labels)):
        if (labels[i] == 'GOP') | (labels[i] == 'realDonaldTrump') | (labels[i] == 'mike_pence'):
            labels[i] = 0
        else:
            labels[i] = 1

    return labels

#%%
class MajorityLabelClassifier():
    """
    A classifier that predicts the mode of training labels
    """
    def __init__(self):
        """
        Initialize your parameter here
        """
        self.m = None
        
    def fit(self, X, y):
        """
        Implement fit by taking training data X and their labels y and finding the mode of y
        i.e. store your learned parameter
        """
        import statistics as st
        self.m = st.mode(y)
    
    def predict(self, X):
        """
        Implement to give the mode of training labels as a prediction for each data instance in X
        return labels
        """
        return np.array([self.m]  * len(X))

#%%
def learn_classifier(X_train, y_train, kernel):
    """ learns a classifier from the input features and labels using the kernel function supplied
    Inputs:
        X_train: scipy.sparse.csr.csr_matrix: sparse matrix of features, output of create_features()
        y_train: numpy.ndarray(int): dense binary vector of class labels, output of create_labels()
        kernel: str: kernel function to be used with classifier. [linear|poly|rbf|sigmoid]
    Outputs:
        sklearn.svm.SVC: classifier learnt from data
    """  
    classif = sklearn.svm.SVC(kernel=kernel, verbose=False)
    classif.fit(X_train, y_train)
    return classif

#%%
def evaluate_classifier(classifier, X_validation, y_validation):
    """ evaluates a classifier based on a supplied validation data
    Inputs:
        classifier: sklearn.svm.SVC: classifer to evaluate
        X_validation: scipy.sparse.csr.csr_matrix: sparse matrix of features
        y_validation: numpy.ndarray(int): dense binary vector of class labels
    Outputs:
        double: accuracy of classifier on the validation data
    """
    return sklearn.metrics.accuracy_score(y_validation, classifier.predict(X_validation))

#%%
def classify_tweets(tfidf, classifier, unlabeled_tweets):
    """ predicts class labels for raw tweet text
    Inputs:
        tfidf: sklearn.feature_extraction.text.TfidfVectorizer: the TfidfVectorizer object used on training data
        classifier: sklearn.svm.SVC: classifier learned
        unlabeled_tweets: pd.DataFrame: tweets read from tweets_test.csv
    Outputs:
        numpy.ndarray(int): dense binary vector of class labels for unlabeled tweets
    """
    return classifier.predict(tfidf.transform(process_all(unlabeled_tweets)['text'])).astype(int)
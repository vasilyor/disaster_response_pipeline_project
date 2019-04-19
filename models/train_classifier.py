# import libraries
import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import FunctionTransformer
import pickle


def load_data(database_filepath):
    
    """ 
    Function for Loading the Data
    
    input:
    	database_filepath: path/filename to SQL database
    	        
    output:
		X: feature variable
		Y: target variable
		category_names:  36 individual category used for clasification
    """

    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('disaster_messages', engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = Y.columns

    return X, Y, category_names


def tokenize(text):

    """ 
    Function for Clean and Tokanize text 
    
    input:
    	text: the original text data
    	        
    output:
		clean_tokens: clean text and tokenize / initiate lemmatizer
    """
    
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    
    """ 
    Function for Building Model.
    Creating pipeline and using grid search to find better parameters.
    
    input:
        None
    output: 
        Scikit model (after the use of GridSearch)
    """

    # build pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
	
    #with more parameters it might take really long time to train the model based on the compute power you have
	#also the pickle model file can grow big 
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        #'vect__max_df': (0.7, 1.0),
        #'vect__max_features': (None, 5000, 10000),
        'clf__estimator__n_estimators': [50, 100]
    }
    
    cv = GridSearchCV(pipeline, parameters)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    
    """ 
    Function for evaluating the Model. 
    
    input:
        model: Scikit model
        X_test: the test data set
        Y_test: the set of labels to all the data in x_test
        category_names:  individual category used for clasification

    output: 
        Print Classification Report (Report the f1 score, precision and recall)
    """

    y_pred = model.predict(X_test)
    print(classification_report(Y_test.iloc[:,1:].values, np.array([x[1:] for x in y_pred]), target_names=categories))

def save_model(model, model_filepath):
    
    """ 
    Export your model as a pickle file
    
    input:
    	model: name of the model which will be saved in a pickled representation
        model_filepath: filepath of the pickle file
    	        
    output:
        serializied {model}.pkl file
    """
    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

#
#https://bbengfort.github.io/tutorials/2016/05/19/text-classification-nltk-sckit-learn.html
#
#https://gist.github.com/bbengfort/044682e76def583a12e6c09209c664a1


import os
import time
import string
import pickle

from operator import itemgetter

from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
#https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report as clsr
#
from sklearn.feature_extraction.text import TfidfVectorizer
# for ngram
from sklearn.feature_extraction.text import CountVectorizer

# new link
#from sklearn.cross_validation import train_test_split as tts
from sklearn.model_selection import train_test_split as tts


def timeit(func):
    """
    Simple timing decorator
    """
    def wrapper(*args, **kwargs):
        start  = time.time()
        result = func(*args, **kwargs)
        delta  = time.time() - start
        return result, delta
    return wrapper


def identity(arg):
    """
    Simple identity function works as a passthrough.
    """
    return arg


class NLTKPreprocessor(BaseEstimator, TransformerMixin):
    """
    Transforms input data by using NLTK tokenization, lemmatization, and
    other normalization and filtering techniques.
    """

    def __init__(self, stopwords=None, punct=None, lower=True, strip=True):
        """
        Instantiates the preprocessor, which make load corpora, models, or do
        other time-intenstive NLTK data loading.
        """
        self.lower      = lower
        self.strip      = strip
        self.stopwords  = set(stopwords) if stopwords else set(sw.words('english'))
        self.punct      = set(punct) if punct else set(string.punctuation)
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        """
        Fit simply returns self, no other information is needed.
        """
        return self

    def inverse_transform(self, X):
        """
        No inverse transformation
        """
        return X

    def transform(self, X):
        """
        Actually runs the preprocessing on each document.
        """
        return [
            list(self.tokenize(doc)) for doc in X
        ]

    def tokenize(self, document):
        """
        Returns a normalized, lemmatized list of tokens from a document by
        applying segmentation (breaking into sentences), then word/punctuation
        tokenization, and finally part of speech tagging. It uses the part of
        speech tags to look up the lemma in WordNet, and returns the lowercase
        version of all the words, removing stopwords and punctuation.
        """
        # Break the document into sentences
        for sent in sent_tokenize(document):
            # Break the sentence into part of speech tagged tokens
            for token, tag in pos_tag(wordpunct_tokenize(sent)):
                # Apply preprocessing to the token
                token = token.lower() if self.lower else token
                token = token.strip() if self.strip else token
                token = token.strip('_') if self.strip else token
                token = token.strip('*') if self.strip else token

                # If punctuation or stopword, ignore token and continue
                if token in self.stopwords or all(char in self.punct for char in token):
                    continue

                #Lemmatize the token and yield
                lemma = self.lemmatize(token, tag)
                yield lemma

    def lemmatize(self, token, tag):
        """
        Converts the Penn Treebank tag to a WordNet POS tag, then uses that
        tag to perform much more accurate WordNet lemmatization.
        """
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(tag[0], wn.NOUN)

        return self.lemmatizer.lemmatize(token, tag)



@timeit

# SVM cannot calculate weight, not sure why
#def build_and_evaluate(X, y, classifier=SVC(kernel='linear'), outpath=None, verbose=True):
#def build_and_evaluate(X, y, classifier=SVC(kernel='sigmoid'), outpath=None, verbose=True):
#def build_and_evaluate(X, y, classifier=LogisticRegression, outpath=None, verbose=True):
def build_and_evaluate(X, y, classifier=SGDClassifier, outpath=None, verbose=True):
    """
    Builds a classifer for the given list of documents and targets in two
    stages: the first does a train/test split and prints a classifier report,
    the second rebuilds the model on the entire corpus and returns it for
    operationalization.
    X: a list or iterable of raw strings, each representing a document.
    y: a list or iterable of labels, which will be label encoded.
    Can specify the classifier to build with: if a class is specified then
    this will build the model with the Scikit-Learn defaults, if an instance
    is given, then it will be used directly in the build pipeline.
    If outpath is given, this function will write the model as a pickle.
    If verbose, this function will print out information to the command line.
    """

    @timeit
    def build(classifier, X, y=None):
        """
        Inner build function that builds a single model.
        """
        if isinstance(classifier, type):
            classifier = classifier()

        # using ngram as feature
        #using logistic regression
        # error AttributeError: 'list' object has no attribute 'lower'
        '''
        model = Pipeline([
            ('preprocessor', NLTKPreprocessor()),
            #('vectorizer', CountVectorizer(tokenizer=identity,  preprocessor=None, lowercase=False, analyzer='word', ngram_range=(1, 3))),
            ('vectorizer', CountVectorizer(ngram_range=(1, 3))),
            ('classifier', classifier),
        ])'''
        model = Pipeline([
            ('vectorizer', CountVectorizer(ngram_range=(1, 3))),
            ('classifier', classifier),
        ])

        '''
        model = Pipeline([
            ('preprocessor', NLTKPreprocessor()),
            ('vectorizer', TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False)),
            ('classifier', classifier),
        ])
        '''

        model.fit(X, y)
        return model

    # Label encode the targets
    # data input order 
    # neg first then pos 
    # so here label seqeuncing 
    # neg become 0 and pos becomes 1
    labels = LabelEncoder()
    y = labels.fit_transform(y)


    # for debug
    index = 0
    for ele in y:
        if index >= 422000 and index < 42000:
            print('{}: {}'.format(index, ele))
        index+=1

    # Begin evaluation
    if verbose: print("Building for evaluation")
    #X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2)

    #using test set directly snice do not care evaluation result
    X_train = X
    X_test = X
    y_train = y
    y_test = y

    model, secs = build(classifier, X_train, y_train)

    if verbose: print("Evaluation model fit in {:0.3f} seconds".format(secs))
    if verbose: print("Classification Report:\n")

    y_pred = model.predict(X_test)
    print(clsr(y_test, y_pred, target_names=labels.classes_))

    if verbose: print("Building complete model and saving ...")
    model, secs = build(classifier, X, y)
    model.labels_ = labels

    if verbose: print("Complete model fit in {:0.3f} seconds".format(secs))

    if outpath:
        with open(outpath, 'wb') as f:
            pickle.dump(model, f)

        print("Model written out to {}".format(outpath))

    return model


def show_most_informative_features(model, text=None, n=20):
    """
    Accepts a Pipeline with a classifer and a TfidfVectorizer and computes
    the n most informative features of the model. If text is given, then will
    compute the most informative features for classifying that text.
    Note that this function will only work on linear models with coefs_
    """
    # Extract the vectorizer and the classifier from the pipeline
    vectorizer = model.named_steps['vectorizer']
    classifier = model.named_steps['classifier']

    # Check to make sure that we can perform this computation
    if not hasattr(classifier, 'coef_'):
        raise TypeError(
            "Cannot compute most informative features on {} model.".format(
                classifier.__class__.__name__
            )
        )

    if text is not None:
        # Compute the coefficients for the text
        tvec = model.transform([text]).toarray()
    else:
        # Otherwise simply use the coefficients
        tvec = classifier.coef_

    # for debug
    # this will oupit all features
    #https://wizardforcel.gitbooks.io/scipycon-2018-sklearn-tut/content/11.html
    '''
    print('number of features {}'.format(len(vectorizer.get_feature_names())))
    index = 0
    for ele in vectorizer.get_feature_names():
        if index < 200:
            print('{}: {}'.format(index, ele))
        index+=1
    '''


    # Zip the feature names with the coefs and sort
    coefs = sorted(
        zip(tvec[0], vectorizer.get_feature_names()),
        key=itemgetter(0), reverse=True
    )

    topn  = zip(coefs[:n], coefs[:-(n+1):-1])

    # Create the output string to return
    output = []

    # If text, add the predicted value to the output.
    if text is not None:
        output.append("\"{}\"".format(text))
        output.append("Classified as: {}".format(model.predict([text])))
        output.append("")

    # Create two columns with most negative and most positive features.
    for (cp, fnp), (cn, fnn) in topn:
        output.append(
            "{:0.4f}\t{: >15}\t{:0.4f}\t{: >15}".format(cp, fnp, cn, fnn)
        )

    with open('feature.txt', 'w', encoding='utf-8') as fout:
        for item in output:
            fout.write(item + '\r\n');



    return "\n".join(output)


if __name__ == "__main__":
    PATH = "model.pickle"

    '''
    if not os.path.exists(PATH):
        # Time to build the model
        from nltk.corpus import movie_reviews as reviews

        X = [reviews.raw(fileid) for fileid in reviews.fileids()]
        y = [reviews.categories(fileid)[0] for fileid in reviews.fileids()]


    

        model = build_and_evaluate(X,y, outpath=PATH)

    else:
        with open(PATH, 'rb') as f:
            model = pickle.load(f)
    '''

    if not os.path.exists(PATH):
        # using subset to test . and it still working
    

        #routine1
        # label output 0  or 1
        # pos 1 and negative 0

        # data input order 
        # neg first then pos 
        '''
        X = []
        y = []
        from nltk.corpus import movie_reviews as reviews
        index = 0
        for ele in reviews.fileids():
            if index < 200:
                #print('{}: {}'.format(index, ele))
                X.append(reviews.raw(ele))
            if index >=1600 and index <= 1800:
                X.append(reviews.raw(ele))
            index+=1

        index = 0
        for ele in reviews.fileids():
            if index < 200:
                print('{}: {}'.format(index, reviews.categories(ele)[0]))
                y.append(reviews.categories(ele)[0])

            if index >=1600 and index <= 1800:
                print('{}: {}'.format(index, reviews.categories(ele)[0]))
                y.append(reviews.categories(ele)[0])
            index+=1
        '''



        # file comes first 
        # so label as 
        # pos : 0
        # neg: 1
        # so negative coefficient means important features for files
        X = []
        y = []

        dedup = set()
        # for speed up test
        #with open('files_domain_training_answer_temp.tsv', 'r', encoding='utf-8') as fin:
        #with open('files_domain_training_contexual_answer_07162020v1.tsv', 'r', encoding='utf-8') as fin:
        with open('files_domain_training_contexual_answer.tsv', 'r', encoding='utf-8') as fin: 
            for line in fin:
                arr = line.split('\t')

                # remove ending of line character
                arr[2] = arr[2].strip()
                arr[3] = arr[3].strip()

                # for debug
                #print(arr[2])
                #print(arr[3])

                X.append(arr[2])

                if arr[3] != 'files':
                    # for debug
                    #print('not_files')
                    y.append('not_files')

                    dedup.add('not_files')
                else:
                    # for debug
                    #print(arr[3])
                    y.append(arr[3])

                    dedup.add(arr[3])

        print('-I-: x {}, y {}'.format(len(X),len(y)))
        print('-I-: y label : {}'.format(len(dedup)))

        model = build_and_evaluate(X,y, outpath=PATH)

        with open(PATH, 'rb') as f:
            model = pickle.load(f)
        print(show_most_informative_features(model))
    else:
        print('model exist so loading directly {}'.format(PATH))
        with open(PATH, 'rb') as f:
            model = pickle.load(f)
        print(show_most_informative_features(model, n=200))
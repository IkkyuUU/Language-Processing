# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 10:23:32 2015

"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline


def PreProcess(dataset, stops):
    
    UselessWords = set(['his', 'him', 'mine', 'our', 'you', 'your', 'its', 'she', 'her', 
                    'they', 'them', 'and', 'too', 'either', 'the', 'for', 'before', 
                    'after', 'this', 'that', 'those', 'these', 'here', 'there',
                    'are', 'was', 'were', 'have', 'did', 'done', 'does', 'do', 'has', 'had'
                    'can', 'could', 'shall', 'should', 'will', 'would', 'itself',
                    'yourself', 'yourselves', 'himself', 'herself', 'themselves', 'myself',
                    'wrote', 'write', 'written','wrapper', 'wrap', 'woudnt',
                    'world', 'workday', 'word', 'wood', 'women', 'woman', 
                    'within', 'wit', 'witch', 
                    'wire', 'wipe', 'winxp', 'winter', 'winston', 'winng', 
                    'wine', 'window', 'wind', 'winc', 'winamp', 'wifi', 'wife' ])
    
    cleanData = list()
    
    stemmer = PorterStemmer()
    
    for line in dataset:
        
        line =re.sub(r'<.*?>',' ',line)#replace chars that are not letters or numbers with a space
        line =re.sub(r'http://.*? ', ' ', line)
        line = re.sub('www.*? ', ' ', line)
        line = re.sub('&quot;|&#39;', '\'', line)
        line = re.sub('&.*?;', ' ', line)
        line = re.sub('didn\'t|didnt', 'did not', line)
        line = re.sub('don\'t|dont', 'do not', line)
        line = re.sub('doesn\'t|doesnt', 'does not', line)
        line = re.sub('haven\'t|havent', 'have not', line)
        line = re.sub('hasn\'t|hasnt', 'has not', line)
        line = re.sub('hadn\'t|hadnt', 'had not', line)
        line = re.sub('won\'t|wont', 'will not', line)
        line = re.sub('wouldn\'t|wouldnt', 'would not', line)
        line = re.sub('can\'t|cannot|cant', 'can not', line)
        line = re.sub('couldn\'t|couldnt', 'could not', line)
        line = re.sub('shouldn\'t|shouldnt', 'should not', line)
        line = re.sub('isn\'t|isnt', 'is not', line)
        line = re.sub('aren\'t', 'are not', line)
        line = re.sub('wasn\'t|wasnt', 'was not', line)
        line = re.sub('weren\'t', 'were not', line)
        line = re.sub('ain\'t|aint', 'be not', line)
        line = re.sub('I\'d', 'I would', line)
        line = re.sub('they\'d', 'they would', line)
        line = re.sub('you\'d', 'you would', line)
        line = re.sub('we\'d', 'we would', line)
        line = re.sub('he\'d', 'he would', line)
        line = re.sub('she\'d', 'she would', line)
        line = re.sub('\'s', ' is', line)
        line = re.sub('\'re', ' are', line)
        line = re.sub('\'ve', ' have', line)
        line = re.sub('\'ll', ' will', line)
        line = re.sub('\'m', ' am', line)
        line = re.sub('[^a-z\d]',' ',line)#replace chars that are not letters or numbers with a space
        line = re.sub(' +',' ',line).strip()#remove duplicate spaces
        line = re.sub('not ', 'not', line)
        line = re.sub('rarely |barely |hardly |merely |never ', 'not', line)
        
        words = line.split(' ')
        
        recombined = []
                
        for word in words:
            if bool(re.search('\d', word)) == True:
                continue
            if len(word) == 1:
                continue
            if word in UselessWords:
                continue
            
            recombined.append(stemmer.stem(word))
        
        cleanData.append(' '.join(recombined))
    
    return cleanData
    


#read the reviews and their polarities from a given file
def loadTrainData(fname):
    reviews=[]
    labels=[]
    f=open(fname)
    for line in f:
        review,rating=line.strip().split('\t') 
        low_review = review.lower() 
        reviews.append(low_review)
        labels.append(int(rating))
    f.close()
    return reviews,labels

def loadTestData(fname):
    reviews=[]
    f=open(fname)
    for line in f:
        review=line.strip() 
        low_review = review.lower()   
        reviews.append(low_review)
    f.close()
    return reviews

rev_train,labels_train=loadTrainData('training.txt')
rev_test=loadTestData('reviews_2.txt')

stops = set(stopwords.words('english'))
newtrain = PreProcess(rev_train, stops)

#testoutput = open('testout.txt', 'w')
#for line in newtrain:
#    testoutput.write(line + '\n')
#testoutput.close()

newtest = PreProcess(rev_test, stops)


#Build a counter based on the training dataset
counter = CountVectorizer(max_df=0.5, ngram_range=(1, 2), binary=True)
counter.fit(newtrain)

#count the number of times each term appears in a document and transform each doc into a count vector
counts_train = counter.transform(newtrain)#transform the training data
counts_test = counter.transform(newtest)#transform the testing data

transformer = TfidfTransformer()
transformer.fit(counts_train)

trans_train = transformer.transform(counts_train)
trans_test = transformer.transform(counts_test)

#pick classifiers
clf1 = LogisticRegression(C=10)
clf2 = KNeighborsClassifier(n_neighbors=5)
clf3 = MultinomialNB(alpha=0.01)
clf4 = PassiveAggressiveClassifier()
clf5 = RandomForestClassifier(n_estimators=50)
clf6 = SGDClassifier(alpha=0.0001)
clf7 = svm.LinearSVC(C = 0.01)
clf8 = AdaBoostClassifier()

#c_range = np.logspace(0, 4, 10)
#clf1_gs = GridSearchCV(estimator=clf7, param_grid=dict(C=c_range), n_jobs=1)
#clf1_gs.fit(trans_train,labels_train)
#print clf1_gs.best_score_
#print clf1_gs.best_estimator_


#
## uncommenting more parameters will give better exploring power but will
## increase processing time in a combinatorial way
parameters = {
    #'vect__max_df': (0.5, 0.75, 1.0),
    #'vect__max_features': (None, 5000, 10000, 50000),
    #'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    #'tfidf__use_idf': (True, False),
    #'tfidf__norm': ('l1', 'l2'),
    'lr__C': (0.01, 0.05, 1.0, 10, 20),
    'lsvc__C': (0.01, 0.05, 1.0, 10, 20),
    'mnb__alpha': (0.0001, 0.01, 1, 10),
    'sgd__alpha': (0.00001, 0.0001),
    'sgd__penalty': ('l2', 'elasticnet'),
    #'clf__n_iter': (10, 50, 80),
}

#build a voting classifer
eclf = VotingClassifier(estimators=[('lr', clf1), ('mnb', clf3), ('sgd', clf6), ('lsvc', clf7)], voting='hard',weights=[2,2,1,1])

#pipeline = Pipeline([
#    ('vc', CountVectorizer()),
#    ('tfidf', TfidfTransformer()),
#    ('clf', eclf),
#])
#
#clf1_gs = GridSearchCV(eclf, parameters)
#eclf.get_params().keys()
#clf1_gs.fit(trans_train,labels_train)
#print clf1_gs.best_score_
#print clf1_gs.best_estimator_.get_params()
##eclf = VotingClassifier(estimators=[('abc',clf8)], voting='hard') 
#eclf.get_params()

#train all classifier on the same datasets
eclf.fit(trans_train,labels_train)

#use hard voting to predict (majority voting)
pred=eclf.predict(trans_test)


#clf1.fit(trans_train, labels_train)
#pred = clf1.predict(trans_test)

fileOutput = open('predictions.txt', 'w')

for p in pred:
    fileOutput.write(str(p) + '\n')

fileOutput.close()

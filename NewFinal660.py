# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 10:23:32 2015

"""

import re
import HTMLParser
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import svm


rows_to_exclude=[4,77,87,159,209,293,308,311,353,383,388,395,435,437,551,554,572,595,610,624,637,677,701,770,815,826,833,845,872,918,955,966,1071,1140,1156,1170,1174,1277,1280,1285,1337,1338,1341,1346,1353,1384,1396,1399,1451,1452,1464,1471,1515,1522,1527,1539,1560,1683,1992,2293,2363,3399,4252,4849,4861,5350,5362,5783,6110,6237,6363,6419,6499,6576,6756,6935, 48,109,748,968,977,1005,1407,1659,1727,1766,1781,1887]
rows_to_flip=[47,93,101,102,120,123,163,181,194,258,379,404,409,411,444,446,448,480,486,494,502,510,517,531,580,596,598,680,697,763,780,798,834,842,883,963,1087,1106,1155,1161,1250,1251,1252,1376,1380,1398,1442,1514,1643,1668,1684,1718,1822,1835,1951,1976,2116,2141,2176,2214,2242,2268,2289,2324,2431,2518,2580,2680,2766,2818,2864,2977,3125,3246,3269,3343,3352,3397,3423,3429,3441,3465,3495,3535,3662,3673,3754,3846,3929,3942,4047,4155,4364,4414,4438,4634,4660,4681,4908,4980,5001,5041,5062,5080,5177,5186,5274,5281,5322,5449,5457,5610,5656,5662,5803,5821,5829,5897,5910,6001,6017,6024,6115,6149,6151,6207,6227,6286,6308,6320,6352,6423,6463,6512,6514,6571,6597,6639,6645,6678,6683,6704,6796,6803,6828,6834,6857,6876,6904,6930,6959,3665,3272,3300,3355,3389,3396,92,146,46,209,238,288,323,348,416,468,470,471,475,513,708,721,755,786,836,984,1114,1225,1300,1363,1431,1454,1588,1598,1630,1639,1690,1691,1708,1749,1783,1915,1962,1985,2014,2040,2082,2083,2156,2186,2192,2251,2302,2316,2388,2457,2527,2565,2582,2660,2661,2667,2742,2772,2773,2792,2865,2871,2874,2875,2897,2989,3008,3027,3072,3080,3091,3102,3103,3123,3160,3222,3272,3300,3355,3389,3396,3398,3482,3528,3571,3665,3685,3746,3747,3765,3872,3911,3941,4003,4004,4106,4109,4121,4193,4231,4241,4334,4344,4374,4409,4411,4422,4425,4558,4560,4568,4629,4671,4693,4800,4841,4865,4877,4879,4882,4924,4984,5076,5085,5239,5314,5346,5400,5401,5442,5479,5497,5525,5530,5540,5554,5628,5653,5689,5690,5717,5838,5944,5954,5962,5988,6005,6175,6259,6303,6386,6557,6586,6640,6652,6699,6729,6740,6887,394,1030,1042,1073,1108,1229,1279,1310,1321,1331,1358,1369,1407,1470,1494,1509,1529,1690,1774,1855,1864,1971,1981,1988,4091,4819,6043,6496,6511,6647,40,107,335,434,599,706,708,830,835,869,889,915,919,945,990,1014,1176,1282,1507,1759,1778,1790,	1796,	1823,	1841,	2108,	2124,	2129,	2153,	2248,	2315,	2318]

def load_train_data(data_file_path):
    reviews = []
    labels = []
    data_file = open(data_file_path)
    idx=1    
    for line in data_file:
        if idx in rows_to_exclude:
            idx = idx + 1
            continue
        review, rating = line.strip().split('\t')
                
        if idx in rows_to_flip:
            if rating=='0':
                rating = '1'
            else:
                rating='0'
                
        idx = idx + 1
        reviews.append(process_row(review))
        labels.append(int(rating))

    data_file.close()
    return reviews, labels

  
def load_test_data(data_file_path):
    reviews = []
    data_file = open(data_file_path)
    for review in data_file:
        reviews.append(process_row(review))

    data_file.close()
    return reviews


first_pass_good="yes!|not go to waste|off the hook|place rocked|won't be sorry|will try again|will be back"
stopwords_pattern="going to seem|the | the | and | a | all i | i | to | was | of | is | for | in | we | with | that | this | on | you | they | were | so | at | are | be | as | very | our | if | there | their | out | or | when | which | me | from | about | an | some | what | been | up | because | only | by | your | more | can | he | after | other | has | too | she | how | do | then | them | who | did | while | few | any | being | off | before | most | both | down | am | her | its | into | his | should | where | why | same | again | those | once | now | such | these | each | having | until | during | him |,|:|food|chicken|'s|<br>|i'm|i'll|i've|\)|\(|\"|jerk chicken| mex |mexican"
not_grams_meh='not exceptional|just ok|might come back|not much better than|not bad|not the best|good but|would not think|better off|really want like'
not_grams_bad="not think come back|not accept|not acceptable|not anymore|not anything|not best|not bother|not clean|not cooked|not cool|not delicious|not enjoy|not even|not fan|not flavorful|not fresh|not friendly|not go|not going|not good|not great|not happy|not hot|not impressed|not impressive|not like|not love|not much|not nice|not ready|not really|not recommend|not returning|not right|not sure|not taste|not tasty|not think|not waste|not well|not worth|not anything special|not big fan|not blown away|not even bother|not get seated|not go place|not my kind|not worth it|not better|much better choices|would go back but|less than mediocre|not come here|not made well|doubt return|likely pass|not par|will not return|not pleasant|not ever going|tastes better than this|not here"
not_grams_vbad="not come back|not coming back|not go back|not going back|not good all|never go back|never come back|totally unacceptable|never return"
not_grams_good="4 stars|5 stars|not wait go back|not get wrong|not bad|not crowded|not disappoint|not expensive|not greasy|not mind|not regret|go back|come back|coming back|never bad|never disappoint|gets right|my kind|worth it|bar none|i ruin|much better than|not had better|looking forward|come here|worked hard|work hard|never failed"
final_pass="dinner|lunch|pork|coffee|dessert|eggs|restaurant|restaurants|kitchen|yet|thai|dumplings|bartender|burger| really | us | would | ordered | restaurant | order | also | - | got | little | i | try | even | people | definitely | first |experience | went | night | cheese | always | bar | could | minutes | eat | make | staff |sauce|sandwich| 2 | dish |pizza|service| all |menu| meal | friend | \& | s |place"
final_pass2_meh="had better"
progfirstpassgood=re.compile(first_pass_good)
progmeh=re.compile(not_grams_meh)
prog = re.compile(stopwords_pattern)
progdont = re.compile("aren&#39;t|don t |cannot|can't|wouldn't|couldn't|wasn't|won't|wont|don't|dont|don&#39;t|did not|didn't|didnt|did&#39;t|shouldn't")
progpunct = re.compile('[!|\?|\.]')
prognotgramsbad = re.compile(not_grams_bad)
prognotgramsvbad = re.compile(not_grams_vbad)
prognotgramsgood = re.compile(not_grams_good)
progfinalpass = re.compile(final_pass)
progfinalpassmeh = re.compile(final_pass2_meh)
html_parser = HTMLParser.HTMLParser()
def process_row(review) :
    review=review.lower()
    review = html_parser.unescape(review)
    
    review = progfirstpassgood.sub(' good ',review)
    review = progdont.sub('not',review)
    review = re.sub("'d ", ' ',review)
    review = progpunct.sub(' . ',review)
    review = prog.sub(' ', review)
    review = prog.sub(' ', review)

    review = prog.sub(' ', review)
    review = progmeh.sub(' meh ', review)
    review = re.sub(' +', ' ',review).strip()        
    review = prognotgramsvbad.sub(' bad bad ',  review)
    review = prognotgramsbad.sub(' bad ', review)
    review = prognotgramsgood.sub(' good ', review)
    review = progfinalpass.sub(' ', review)
    review = progfinalpassmeh.sub(' meh ', review)   
    return review

    

train_reviews, train_targets = load_train_data('training.txt')
test_reviews = load_test_data('testing.txt')    
#Build a counter based on the training dataset
counter = CountVectorizer(max_df=0.5, ngram_range=(1, 2), binary=True)
counter.fit(train_reviews)

#count the number of times each term appears in a document and transform each doc into a count vector
counts_train = counter.transform(train_reviews)#transform the training data
counts_test = counter.transform(test_reviews)#transform the testing data

transformer = TfidfTransformer()
transformer.fit(counts_train)

trans_train = transformer.transform(counts_train)
trans_test = transformer.transform(counts_test)

#pick classifiers
clf1 = LogisticRegression()
clf2 = KNeighborsClassifier(n_neighbors=5)
clf3 = MultinomialNB()
clf6 = SGDClassifier()
clf7 = svm.LinearSVC()



#build a voting classifer
eclf = VotingClassifier(estimators=[('lr', clf1), ('mnb', clf3), ('sgd', clf6), ('lsvc', clf7)], voting='hard',weights=[2,2,1,1])
#eclf = VotingClassifier(estimators=[('abc',clf8)], voting='hard') 

#train all classifier on the same datasets
eclf.fit(trans_train,train_targets)

#use hard voting to predict (majority voting)
pred=eclf.predict(trans_test)

#MNB_pipeline = Pipeline([('vect', CountVectorizer(ngram_range = (1, 3))), 
#                         ('clf', MultinomialNB(alpha = .01)),
#                        ])
#                        
#KNN_pipeline = Pipeline([('vect', CountVectorizer()), 
#                         ('clf', KNeighborsClassifier(p=2,n_neighbors = 31)),
#                        ])
#                            
#LR_pipeline = Pipeline([('vect', CountVectorizer()), 
#                        ('tfidf', TfidfTransformer(norm = 'l2', use_idf = True, smooth_idf = True, sublinear_tf = True)),
#                        ('clf', LogisticRegression(warm_start=True, dual=False, penalty='l2',solver='newton-cg')),
#                       ])
#                       
#
#VC = VotingClassifier(estimators=[('MNB', MNB_pipeline), ('KNN', KNN_pipeline), ('LR', LR_pipeline)], voting = 'soft', weights = [2, 1, 2]) 
#
#
#VC.fit(train_reviews, train_targets)
#
#test_reviews = load_test_data('testing.txt')
#predicted = VC.predict(test_reviews)

output_file = open("out.txt", "w")
for p in pred:
    output_file.write(str(p) + '\n')
output_file.close()

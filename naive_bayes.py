# CS114 Spring 2019 Homework 3
# Naive Bayes Classifier and Evaluation

import os
import numpy as np
from collections import defaultdict
from tabulate import tabulate

class NaiveBayes():

    def __init__(self):
        self.class_dict = {0: 'neg', 1: 'pos'}
        #self.feature_dict = {0: 'great', 1: 'poor', 2: 'good'}
        self.feature_dict = defaultdict(str)
        self.prior = np.array([0,0])
        self.likelihood = None

        self.total_docs = 0
        self.vocabulary = []
        self.big_doc = defaultdict(list)
        self.feature_count = defaultdict(lambda: defaultdict(int))


    '''
    Trains a multinomial Naive Bayes classifier on a training set.
    Specifically, fills in self.prior and self.likelihood such that:
    self.prior[class] = log(P(class))
    self.likelihood[class][feature] = log(P(feature|class))
    '''
    def train(self, train_set):
        # iterate over training documents
        for root, dirs, files in os.walk(train_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    review = f.read()
                    self.total_docs+=1
                    # collect class counts and feature counts
                    sentiment = os.path.basename(root)
                    for cat, cat_name in self.class_dict.items():
                        if sentiment == cat_name:
                            self.prior[cat]+=1
                            words = review.split()
                            self.big_doc[cat].extend(words)
                            for word in words:
                                self.feature_count[cat][word] +=1
        for key, value in self.big_doc.items():
            self.vocabulary.extend(self.big_doc[key])
        self.vocabulary = list(set(self.vocabulary))

        counter = 0
        for word in self.vocabulary:
            prob_word_in_neg = self.feature_count[0][word]/len(self.big_doc[0])
            prob_word_in_pos = self.feature_count[1][word]/len(self.big_doc[1])
            if word.endswith(('ed','ing','ly')) or (prob_word_in_neg - prob_word_in_pos > prob_word_in_neg/2 and self.feature_count[0][word]>2) or (prob_word_in_pos - prob_word_in_neg > prob_word_in_pos/2 and self.feature_count[1][word]>2):
                self.feature_dict[counter] = word
                counter += 1

        self.prior = np.divide(self.prior, self.total_docs)
        self.prior = np.log(self.prior)
        self.likelihood = np.zeros([len(self.class_dict), len(self.feature_dict)], dtype = 'float64')
        for cat, cat_name in self.class_dict.items():
            for id, feature in self.feature_dict.items():
                self.likelihood[cat][id] = (self.feature_count[cat][feature] + 1)/(len(self.big_doc[cat]) + len(self.vocabulary))
        self.likelihood = np.log(self.likelihood)
        #print(self.likelihood)


    def test(self, dev_set):
        results = defaultdict(dict)
        # iterate over testing documents
        for root, dirs, files in os.walk(dev_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    correct_sentiment = os.path.basename(root)
                    for value, tag in self.class_dict.items():
                        if tag == correct_sentiment:
                            correct_sentiment = value
                    counts = defaultdict(int)
                    test_vector = np.zeros(len(self.feature_dict), dtype='float64')
                    review = f.read()
                    words = review.split()
                    for word in words:
                        counts[word]+=1
                    for id, feature in self.feature_dict.items():
                        test_vector[id] = counts[feature]
                    product = self.likelihood.dot(test_vector)
                    sum = product + self.prior
                    if sum[0]>sum[1]:
                        predicted_sentiment = 0
                    else:
                        predicted_sentiment = 1
                    results[name]['correct'] = correct_sentiment
                    results[name]['predicted'] = predicted_sentiment
        return results

    '''
    Given results, calculates the following:
    Precision, Recall, F1 for each class
    Accuracy overall
    Also, prints evaluation metrics in readable format.
    '''
    def evaluate(self, results):
        # you may find this helpful
        confusion_matrix = np.zeros((2, 2), dtype=int)
        #print(confusion_matrix)
        for file, value in results.items():
                confusion_matrix[value['correct'], value['predicted']]+=1
        #print(confusion_matrix)
        total_test_docs = confusion_matrix[0][0] + confusion_matrix[0][1] + confusion_matrix[1][0] + confusion_matrix[1][1]
        accuracy = round((confusion_matrix[0][0] + confusion_matrix[1][1])/total_test_docs,2)

        #positive class metrics
        precision_neg = round(confusion_matrix[0][0]/(confusion_matrix[0][0] + confusion_matrix[0][1]),2)
        recall_neg = round(confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[1][0]),2)
        f1_neg = round(2*precision_neg*recall_neg/(precision_neg+recall_neg),2)

        # negative class metrics
        precision_pos = round(confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[1][0]), 2)
        recall_pos = round(confusion_matrix[1][1] / (confusion_matrix[0][1] + confusion_matrix[1][1]),2)
        f1_pos = round(2 * precision_pos * recall_pos / (precision_pos + recall_pos),2)


        table = [['category', 'precision','recall','f1'],['pos', precision_pos, recall_pos, f1_pos],['neg',precision_neg,recall_neg,f1_neg]]
        print(tabulate(table))
        print("General accuracy: " + str(accuracy))


if __name__ == '__main__':
    nb = NaiveBayes()
    # make sure these point to the right directories
    nb.train('movie_reviews/train')
    results = nb.test('movie_reviews/dev')
    nb.evaluate(results)

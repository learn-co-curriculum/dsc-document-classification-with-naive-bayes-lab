import numpy as np
import pandas as pd


def bag_it(doc):
            bag = {}
            for word in doc.split():
                bag[word] = bag.get(word, 0) + 1
            return bag


#OOP Version
class nb_doc_cf():
    # __init__(self):


    def fit(self, X, y):
        self.df = pd.concat([X, y], axis=1) 
        self.df.columns = ['text', 'label']


        #Calculate Training Class Word Frequencies
        self.class_word_freq = {} #Will be a nested dictionary of class_i : {word1:freq, word2:freq..., wordn:freq},.... class_m : {}
        self.classes = self.df.label.unique()
        for class_ in self.classes:
            temp_df = self.df[self.df.label==class_]
            bag = {}
            for row in temp_df.index:
                doc = temp_df['text'][row]
                for word in doc.split():
                    bag[word] = bag.get(word, 0) + 1
            self.class_word_freq[class_] = bag
        self.p_classes = dict(self.df.label.value_counts(normalize=True))


        #Calculate size of training vocabulary
        self.vocabulary = set()
        for text in self.df.text:
            for word in text.split():
                self.vocabulary.add(word)
        self.V = len(self.vocabulary)
    def predict(self, doc, return_posteriors=False, use_logs=True):
        bag = bag_it(doc)
        classes = []
        posteriors = []
        for class_ in self.class_word_freq.keys():
            if use_logs:
                p = np.log(self.p_classes[class_])
            else:
                p = self.p_classes[class_]
            for word in bag.keys():
                num = bag[word]+1
                denom = self.class_word_freq[class_].get(word, 0) + self.V
                if use_logs:
                    p += np.log(num/denom)
                else:
                    p *= (num/denom)

            classes.append(class_)
            posteriors.append(p)
        if return_posteriors:
            return posteriors
        else:
	        return classes[np.argmax(posteriors)]

        




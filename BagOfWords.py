import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


# Create the bag of words feature matrix
def Bag_of_words(data):
    count = CountVectorizer(stop_words='english')
    bag_of_words = count.fit_transform(data.tokenized_stem)
    feature_names = count.get_feature_names()


    pd1=pd.DataFrame(bag_of_words.todense(), columns=feature_names)
    
    return pd1
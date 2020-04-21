from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer 
from nltk.corpus import stopwords 
from collections import Counter


class TokenizationStemming():
    
    
    
    
    
    '''
    def __init__(self, previous):
        self.processed_data = previous.processed_data'''
    
    def stem(self,tweets):
        stemmer = PorterStemmer()
        tweets['stemmed_opinion'] =tweets["tokenized_text"].apply(lambda x: [stemmer.stem(y) for y in x])
        return tweets

        

    def tokenize(self,tweets):
        stop_words = set(stopwords.words('english')) 
        
        tweets['tokenized_text']=tweets['opinion'].str.lower()
        #tweets['tokenized_text'] = tweets['opinion'].apply(lambda x: [item for item in x if item not in stop_words])
        tweets['tokenized_text']=tweets['opinion'].apply(word_tokenize)
        tweets['tokenized_text'] = tweets['tokenized_text'].apply(lambda x: [y for y in x if y not in stop_words])
        #tweets['tokenized_text'] = tweets['opinion'].apply(lambda x: x if x not in stop_words)
        return tweets
        
    def word_counter(self,tweets):
        '''for row in tweets.iterrows():
            count = tweets['tokenized_text'].value_counts() 
        print(count)'''
        count=Counter()
        for idx in tweets.index:
            count.update(tweets.loc[idx, "tokenized_text"])

        return count

        
        
    def stemming(self,tweets):
        stemmer = PorterStemmer()
        tweets['tokenized_stem'] =tweets['opinion'].str.lower()
        tweets['tokenized_stem']=tweets['tokenized_stem'].apply(word_tokenize)
        tweets['tokenized_stem'] = tweets['tokenized_stem'].apply(lambda x: [stemmer.stem(y) for y in x])
        tweets['tokenized_stem'] =[" ".join([word for word in article]) for article in tweets['tokenized_stem']]
    # print the first article as a running example
        return tweets
        

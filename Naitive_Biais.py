from sklearn.naive_bayes import MultinomialNB



def Naive_Bais(bow,y): 
    
    model1 = MultinomialNB()

    model1.fit(bow,y) 
    
    
    return model1



from xgboost import XGBClassifier as XGBoostClassifier


def XGBoost_classier(bow,y): #Naive_Bais(bow,tweets):
    
    model1 = XGBClassifier()

    model1.fit(bow,y) # model1.fit(bow,tweets.review)
    
    
    return model1
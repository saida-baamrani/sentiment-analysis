from sklearn.ensemble import RandomForestClassifier


def Random_Forest(bow,y): 
    
    model1 = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)

    model1.fit(bow,y) 
    
    
    return model1
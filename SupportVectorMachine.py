from sklearn import svm

def Support_Vector(bow,y): 
    
    model1 =svm.SVC(gamma='scale', probability=True)

    model1.fit(bow,y) 
    
    
    return model1
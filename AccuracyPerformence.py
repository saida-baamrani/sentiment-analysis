from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from matplotlib.pyplot import *
from sklearn.metrics import mean_absolute_error


def predict(model,bow):
    pred = model.predict(bow)
    return pred
   
def predict_proba(model,bow):
    pred = model.predict_proba(bow)
    return pred    

##AUC AND ACC    
def performance(y_true, pred, color="g", ann=True):
    acc = accuracy_score(y_true, pred[:,1] > 0.5)
    auc = roc_auc_score(y_true, pred[:,1])
    fpr, tpr, thr = roc_curve(y_true, pred[:,1])
    plot(fpr, tpr, color, linewidth="3")
    xlabel("False positive rate")
    ylabel("True positive rate")
    if ann:
        annotate("Acc: %0.2f" % acc, (0.2,0.7), size=14)
        annotate("AUC: %0.2f" % auc, (0.2,0.6), size=14)
        
def mean_abs_err(y_pred,y_true):
     val_mae =mean_absolute_error(y_pred,y_true)
     
     print(val_mae)


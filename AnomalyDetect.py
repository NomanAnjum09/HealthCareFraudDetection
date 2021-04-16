from pycaret.anomaly import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score

data = pd.read_csv('./HealthCare/Train_Inpatientdata.csv',nrows=10000)
out_data = pd.read_csv("./HealthCare/Train_Outpatientdata.csv",nrows=10000)
labels = pd.read_csv('./HealthCare/Train.csv')
y = labels.merge(data)['PotentialFraud'].eq('Yes').mul(1)
out_y = labels.merge(out_data)['PotentialFraud'].eq('Yes').mul(1)


print("Ratio of Anomalies in Inpatient Data",(len(y[y==1])/len(y)))
print("Ratio of Anomalies in Outpatient Data",(len(out_y[out_y==1])/len(out_y)))

in_ano_ratio = len(y[y==1])/len(y)
out_ano_ratio = len(out_y[out_y==1])/len(out_y)


ano1 = setup(data = data)

## Isolation Forest

## Tested Different Varitations of anomaly fraction
## Default 0.05 accuracy 63% but less anomalies were detected
## Actual 0.35  accuracy 52% around 1200 anomalies detected out of 10000

iforest = create_model('iforest',fraction=in_ano_ratio)
ifor_pred = predict_model(iforest,data=data)
y_pred = ifor_pred['Anomaly']
print("Isolation Forest Accuracy: ",accuracy_score(y,y_pred))
print(confusion_matrix(y,y_pred))


## KNN Anomaly Detection

knn = create_model('knn',fraction = in_ano_ratio)
knn_pred = predict_model(knn,data=data) 
knn_pred[knn_pred['Anomaly']==1]

print("KNN Accuracy: ",accuracy_score(y,knn_pred['Anomaly']))
print(confusion_matrix(y,knn_pred['Anomaly']))

## Angle Based Outlier Detection
abod_model = create_model('abod',fraction = in_ano_ratio)
abod_pred = predict_model(abod_model,data=data) 
print("ABOD Accuracy: ",accuracy_score(y,abod_pred['Anomaly']))
print(confusion_matrix(y,abod_pred['Anomaly']))

## Cluster Based
cluster_model = create_model('cluster',fraction = in_ano_ratio)
cluster_pred = predict_model(cluster_model,data=data) 
print("Cluster Accuracy: ",accuracy_score(y,cluster_pred['Anomaly']))
print(confusion_matrix(y,cluster_pred['Anomaly']))

## Stochastic
stochastic_model = create_model('sos',fraction = in_ano_ratio)
sto_pred = predict_model(stochastic_model,data=data)
print("Stochastic Accuracy: ",accuracy_score(y,sto_pred['Anomaly']))
print(confusion_matrix(y,sto_pred['Anomaly']))

## Minimum Covariance
mcd_model = create_model('mcd',fraction = in_ano_ratio)
mcd_pred = predict_model(mcd_model,data=data)
print("Minumum Covariance Accuracy: ",accuracy_score(y,mcd_pred['Anomaly']))
print(confusion_matrix(y,mcd_pred['Anomaly']))

####  Anomaly Detection Deosn't seems to be a good tool for detcting anomalies in this dataset.
####  Either we need to change technique for detection of fraud, or we need to mould our data to some aggregate

# For OutPatient Data


ano1 = setup(data = out_data)
## Isolation Forest
iforest = create_model('iforest',fraction=out_ano_ratio)
ifor_pred = predict_model(iforest,data=out_data)
y_pred = ifor_pred['Anomaly']
print("Isolation Forest Accuracy: ",accuracy_score(y,y_pred))
print(confusion_matrix(y,y_pred))

## KNN Anomaly Detection
knn = create_model('knn',fraction=out_ano_ratio)
knn_pred = predict_model(knn,data=out_data) 
print("KNN Accuracy: ",accuracy_score(y,knn_pred['Anomaly']))
print(confusion_matrix(y,knn_pred['Anomaly']))

## Angle Based Outlier Detection
abod_model = create_model('abod',fraction=out_ano_ratio)
abod_pred = predict_model(abod_model,data=out_data) 
print("ABOD Accuracy: ",accuracy_score(y,abod_pred['Anomaly']))
print(confusion_matrix(y,abod_pred['Anomaly']))

## Cluster Based
cluster_model = create_model('cluster',fraction=out_ano_ratio)
cluster_pred = predict_model(cluster_model,data=out_data) 
print("Cluster Accuracy: ",accuracy_score(y,cluster_pred['Anomaly']))
print(confusion_matrix(y,cluster_pred['Anomaly']))

## Stochastic
stochastic_model = create_model('sos',fraction=out_ano_ratio)
sto_pred = predict_model(stochastic_model,data=out_data)
print("Stochastic Accuracy: ",accuracy_score(y,sto_pred['Anomaly']))
print(confusion_matrix(y,sto_pred['Anomaly']))
## Minimum Corvarince
mcd_model = create_model('mcd',fraction=out_ano_ratio)
mcd_pred = predict_model(mcd_model,data=out_data)
print("Minumum Covariance Accuracy: ",accuracy_score(y,mcd_pred['Anomaly']))
print(confusion_matrix(y,mcd_pred['Anomaly']))








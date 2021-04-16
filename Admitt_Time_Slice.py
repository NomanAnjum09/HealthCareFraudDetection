import pandas as pd
import featuretools as ft
import numpy as np
import composeml as cp
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


ben_data = pd.read_csv("./HealthCare/Train_Beneficiarydata.csv").sample(frac=0.5)
admit_patient_data = pd.read_csv("./HealthCare/Train_Inpatientdata.csv").sample(frac=0.5)
admit_patient_data['ClaimStartDt'] = pd.to_datetime(admit_patient_data['ClaimStartDt'], errors='coerce')
labels = pd.read_csv("./HealthCare/Train.csv")
admitted = labels.merge(admit_patient_data)


print(admitted)
print("")

## To Get Number of claims By a Provider within Specific Time
admitted['ClaimCount'] = 1

def generate_time_slices(Slice = 'M'):
    gr_Df = admitted.groupby('Provider').resample(Slice, on='ClaimStartDt').sum().reset_index()
    gr_Df = gr_Df.merge(labels)
    gr_Df.drop(['ClmProcedureCode_1','ClmProcedureCode_2','ClmProcedureCode_3','ClmProcedureCode_4','ClmProcedureCode_5','ClmProcedureCode_6'] , axis=1).to_csv("admit_time_slice_"+Slice+".csv")
    return gr_Df
# Paramters
##  'W'  : weekly frequency
##  'M'  : month end frequency
##  'SM' : semi-month end frequency (15th and end of month)
##  'Q'  : quarter end frequency

sliced = generate_time_slices('M')


### Non Fraud Claims
print("Non Fraudulent Claims Grouped With Respect To One Month")
print(sliced[sliced['PotentialFraud']== 'No'])
print("Maximum Claims By Non Fraudulent in a month :",sliced[sliced['PotentialFraud']== 'No']['ClaimCount'].max())
print("Minimum Claims By Non Fraudulent in a month :",sliced[sliced['PotentialFraud']== 'No']['ClaimCount'].min())
print("STD Claims By Non Fraudulent in a month :",sliced[sliced['PotentialFraud']== 'No']['ClaimCount'].std())
print("Mean Claims By Non Fraudulent in a month :",sliced[sliced['PotentialFraud']== 'No']['ClaimCount'].mean())
print("")




### Fraud Claims
print("Fraudulent Claims Grouped With Respect To One Month")
print(sliced[sliced['PotentialFraud']== 'Yes'])
print("Maximum Claims By Fraudulent in a month :",sliced[sliced['PotentialFraud']== 'Yes']['ClaimCount'].max())
print("Minimum Claims By Fraudulent in a month :",sliced[sliced['PotentialFraud']== 'Yes']['ClaimCount'].min())
print("STD Claims By Fraudulent in a month :",sliced[sliced['PotentialFraud']== 'Yes']['ClaimCount'].std())
print("Mean Claims By Fraudulent in a month :",sliced[sliced['PotentialFraud']== 'Yes']['ClaimCount'].mean())
#### Though fraudulent claims are higher as compared to no fraudulent within a month. However it is very less as compared to checkup claims

print("")
print("")

## Analyzing Reimbursed AMount By Fraudulent Provider
print("Maximum Reimbursed Amount By Fraudulent Provider: ",sliced[sliced['PotentialFraud']== 'Yes']['InscClaimAmtReimbursed'].max())
print("Minimum Reimbursed Amount By Fraudulent Provider: ",sliced[sliced['PotentialFraud']== 'Yes']['InscClaimAmtReimbursed'].min())
print("Mean Reimbursed Amount By Fraudulent Provider: ",sliced[sliced['PotentialFraud']== 'Yes']['InscClaimAmtReimbursed'].mean())
print("STD Reimbursed Amount By Fraudulent Provider: ",sliced[sliced['PotentialFraud']== 'Yes']['InscClaimAmtReimbursed'].std())

print("")

## Analyzing Reimbursed AMount By Non Fraudulent Provider
print("Maximum Reimbursed Amount By Non Fraudulent Provider: ",sliced[sliced['PotentialFraud']== 'No']['InscClaimAmtReimbursed'].max())
print("Minimum Reimbursed Amount By Non Fraudulent Provider: ",sliced[sliced['PotentialFraud']== 'No']['InscClaimAmtReimbursed'].min())
print("Mean Reimbursed Amount By Non Fraudulent Provider: ",sliced[sliced['PotentialFraud']== 'No']['InscClaimAmtReimbursed'].mean())
print("STD Reimbursed Amount By Non Fraudulent Provider: ",sliced[sliced['PotentialFraud']== 'No']['InscClaimAmtReimbursed'].std())

### Very clearly large claims means large amount reimbursed


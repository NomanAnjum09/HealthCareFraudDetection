import pandas as pd
import featuretools as ft
import numpy as np
import composeml as cp

ben_data = pd.read_csv("./HealthCare/Train_Beneficiarydata.csv").sample(frac=0.5)
admit_patient_data = pd.read_csv("./HealthCare/Train_Inpatientdata.csv").sample(frac=0.5)
checked_patient_data = pd.read_csv("./HealthCare/Train_Outpatientdata.csv").sample(frac=0.5)
labels = pd.read_csv("./HealthCare/Train.csv")

print("Beneciary Data: ")
print(ben_data.head())
print("")

print("Admitted Patients Data: ")
print(admit_patient_data.head(5))
print("")

print("Checked Patients Data: ")
print(checked_patient_data.head(5))
print("")

print("Labels: ")
print(labels.head(5))
print("")

#Little Preprocessing
print("Null Values in Beneficiary Data Columns: ")
print(ben_data.isna().sum())  # No null
print("")

print("Null Values In Admitted Patients Data Columns: ")
print(len(admit_patient_data))
print(admit_patient_data.isna().sum()) # Only 9 patients went for proc 5 and none for 6. Drop these.
print("")

print("Null Values in Checked Patients Data COlumns: ")
print(len(checked_patient_data))
print(checked_patient_data.isna().sum())
### Very much sparse data. Drop  proc code 3,4,5,6
print("")


to_drop = ['ClmProcedureCode_5','ClmProcedureCode_6']
admit_patient_data = admit_patient_data.drop(to_drop,axis=1)
to_drop = ['ClmProcedureCode_3','ClmProcedureCode_4','ClmProcedureCode_5','ClmProcedureCode_6']
checked_patient_data = checked_patient_data.drop(to_drop,axis=1)

## Feature Synthesis

entities = {
    "Benficiary_Data" : (ben_data,"BeneID"),
    "checked_patient_Data" : (checked_patient_data, "ClaimID"),
    "admitted_patient_Data" : (admit_patient_data,"ClaimID"),
    "labels" : (labels,"Provider")
}
Relationship = {("Benficiary_Data","BeneID","checked_patient_Data","BeneID"),
                ("Benficiary_Data","BeneID","admitted_patient_Data","BeneID"),
                ("labels","Provider","checked_patient_Data","Provider"),
                ("labels","Provider","admitted_patient_Data","Provider")
               }
print("Feature Matrix With Beneficiary as Base: ")
feature_matrix_ben, feature_df_ben = ft.dfs(
    entities=entities,
    relationships=Relationship,
    target_entity= 'Benficiary_Data',
    verbose=True
                    
)
print(feature_matrix_ben)
print("\n Features Created\n",feature_df_ben)

print("Maximum Times a beneficicary claimed to be admitted : {}".format(feature_matrix_ben['COUNT(admitted_patient_Data)'].max()))
print("Beneficiaries Who claimed for admission more than 4 times : {}".format(feature_matrix_ben[feature_matrix_ben['COUNT(admitted_patient_Data)'] >4]))


print("Maximum Times a beneficicary claimed to be checked : {}".format(feature_matrix_ben['COUNT(checked_patient_Data)'].max()))
print("Beneficiaries Who claimed for admission more than 15 times : {}".format(feature_matrix_ben[feature_matrix_ben['COUNT(checked_patient_Data)'] >4]))

print("Feature Matrix With Beneficiary as Base: ")
feature_matrix_provider, feature_df_provider = ft.dfs(
    entities=entities,
    relationships=Relationship,
    target_entity= 'labels',
    verbose=True
                    
)

print("Feature Matrix Of Provider: ")
print(feature_df_provider)


## This Feature Synthesis with beneficiary as Base, will help us get further insight 


# print("\nFeature Matrix With Checked Beneficiary as Base: ")
# feature_matrix_check, feature_df_check = ft.dfs(
#     entities=entities,
#     relationships=Relationship,
#     target_entity= 'checked_patient_Data',
#     verbose=True
                    
# )

# print(feature_matrix_check)
# print("\n Features Created\n",feature_df_check)

# print("\nFeature Matrix With Admitted Beneficiary as Base: ")
# feature_matrix_adm, feature_df_adm = ft.dfs(
#     entities=entities,
#     relationships=Relationship,
#     target_entity= 'admitted_patient_Data',
#     verbose=True
                    
# )

# print(feature_matrix_adm)
# print("\n Features Created\n",feature_df_adm)

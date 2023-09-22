import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

#%% Import CSV file
df_merged = pd.read_csv("file location")

#Cross products features 
df_cpf = df_merged.copy()
df_cpf = df_cpf.drop(["Abstract","Update_Date","Number_Versions"],axis=1)

#Transforms any subject to one and any other to 0
def return_total_amount_subject(value):
    if value == 0:
        return 1
    else:
        return 0

df_cpf['Subject'] = df_cpf['Subject'].apply(return_total_amount_subject)

#Transforms topic form 0 to 9 to one and any other to 0
def return_total_amount(value):
    if value in range(10):
        return 1
    else:
        return 0

df_cpf['Topic_Id'] = df_cpf['Topic_Id'].apply(return_total_amount)

#one-hot encoding
df_cpf = pd.get_dummies(df_cpf)

#generate interaction features 
pf = PolynomialFeatures(degree=2, \
    interaction_only=True, include_bias=False).\
        fit(df_cpf[['Subject','Topic_Id']])
int_feat = pf.transform(df_cpf[['Subject', \
    'Topic_Id']])

# convert the generated interaction feature array to a dataframe
#subject 0
sub_x_topic = pd.DataFrame(int_feat, \
    columns=['Subject','Topic_Id', 'SxTopic'])

# Calculate the mean value
print(sub_x_topic['SxTopic'].mean(0))
import pandas as pd
import numpy as np
from top2vec import Top2Vec

#%% Import CSV file
df_csv = pd.read_csv("location of the file")

# Drop unnecessary columns
df_csv = df_csv.drop(["Id", "Submitter", "Authors", "Title", "Comments", "Journal-Ref", "DOI", "Report-No", "License", "Authors_Parsed"],axis=1)
df_csv = df_csv.drop(df_csv.columns[(0)],axis=1)

#%%Check for NaN values in all columns
category_list = ["Categories","Abstract","Update_Date","Versions"]
for x in range(4):
    check_nan = df_csv[category_list[x]].isnull().values.any()
    print(check_nan)

#Replace values of category with subject from arxiv.org
#%%Feature encoding was used by a creating a custom function

list_physics = ["astro-ph","cond-mat","gr-qc","hep-ex","hep-lat","hep-ph","hep-th","math-ph","nlin","nucl-ex","nucl-th","physics","quant-ph"]
list_math = ["math"]
list_cs = ["cs"]
list_bio = ["q-bio"]
list_fin = ["q-fin"]
list_stat = ["stat"]
list_eess = ["eess"]
list_econ = ["econ"]

def return_subject(value):
    
    current_value = value.split(" ", -1)
    current_topic = current_value[0].split(".",-1)
    if len(current_topic) > 1:
        current_topic.pop()

    if current_topic[0] in list_physics:
        #Physics
        return 0
    elif current_topic[0] in list_math:
        #Mathematics
        return 1
    elif current_topic[0] in list_cs:
        #Computer Science
        return 2
    elif current_topic[0] in list_bio:
        #Quantitative Biology
        return 3
    elif current_topic[0] in list_fin:
        #Quantitative Finance
        return 4
    elif current_topic[0] in list_stat:
        #Statistics
        return 5
    elif current_topic[0] in list_eess:
        #Electrical Engineering and Systems Science
        return 6
    else:
        #Economics
        return 7

df_csv['Subject'] = df_csv['Categories'].apply(return_subject)

#Drop cateregories as it is replaced with subject
df_csv = df_csv.drop(["Categories"],axis=1)

#add a column with the amount of verions of the doc
def return_total_amount(value):
    current_value = str(value).split("}, {", -1)
    total_len = len(current_value)
    
    return total_len

df_csv['Number_Versions'] = df_csv['Versions'].apply(return_total_amount)

#Drop versions as it is replaced with number_versions
df_csv = df_csv.drop(["Versions"],axis=1)

#Top2Vec
#%%Set a managable sample
df_csv_sample = df_csv.sample(n=100000)
#Remove duplicate values from the sample
df_csv_sample = df_csv_sample.drop_duplicates(subset=['Abstract'])
#Convert update_date to datetime type
df_csv_sample['Update_Date'] = pd.to_datetime(df_csv_sample['Update_Date'])
#Convert abstract to string type
df_csv_sample['Abstract'] = df_csv_sample['Abstract'].astype("string")
#%%Verify that no duplicate is left
any(df_csv_sample["Abstract"].duplicated())

#%%run Top2Vec model
model = Top2Vec(list(df_csv_sample["Abstract"]))

#query the model
topic_sizes, topic_nums = model.get_topic_sizes()

#%%shows the documents with the highest probability and the row related to the doc
documents, document_scores, document_ids = model.search_documents_by_topic(topic_num=2, num_docs=5)
for doc, score, doc_id in zip(documents, document_scores, document_ids):
    print(f"Document: {doc_id}, Score: {score}")
    print("-----------")
    print(doc)
    print("-----------")
    print(df_csv.loc[df_csv['Abstract'] == doc])
    print("-----------")
    print()

#%%Create a df with the model infomation
columns_df = ["Topic_Id","Abstract"]
df_output = pd.DataFrame(columns=columns_df)

for (topic_id, topic_documents_number) in zip(topic_nums, topic_sizes):   
    documents, document_scores, document_ids = model.search_documents_by_topic(topic_num=topic_id, num_docs=topic_documents_number)
    for doc in documents:
        df_output.loc[len(df_output.index)] = [topic_id,doc]

#Merged and exported the file
df_merged = pd.merge(df_csv_sample,df_output, on="Abstract")
df_merged.to_csv("df_merged_data.csv")
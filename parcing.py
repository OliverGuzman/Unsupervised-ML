import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

#%%import json file, randomize it and parse it.
df = pd.read_json("location of the file", lines= True)
set_1, set_2, set_3 = np.split(df.sample(frac=1), [int(0.4*len(df)), int(0.7*len(df))])

# Convert numpy array to list
list_set_1 = set_1.values.tolist()

# Serialize to JSON
json_set_1 = json.dumps(list_set_1)

# Writing to sample.json
jsonFile = open("data_set_1.json", "w")
jsonFile.write(json_set_1)
jsonFile.close()

#import JSON file into pandas DataFrame and add columns titles
cols = ["Id", "Submitter", "Authors", "Title", "Comments", "Journal-Ref", "DOI", "Report-No", "Categories", "License", "Abstract", "Versions", "Update_Date", "Authors_Parsed"]
df = pd.read_json(f'/kaggle/working/data_set_1.json')
df.columns = cols

#To csv
df.to_csv(f"data_set_1.csv")
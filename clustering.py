import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import mixture

#%% Import CSV file
df_merged = pd.read_csv("file location")

#Convert update_date to datetime type
df_merged['Update_Date'] = pd.to_datetime(df_merged['Update_Date'])
#Convert abstract to string type
df_merged['Abstract'] = df_merged['Abstract'].astype("string")
#Remove Id of row
df_merged = df_merged.drop(df_merged.columns[(0)],axis=1)

#Filter of the dataframe
df_merged_1 = df_merged[df_merged['Topic_Id'] < 7].loc[df_merged['Update_Date'] > "2000-01-01"].loc[df_merged['Subject'] <6]

#GMM
#%%Additionally, there is a for loop for checking which combination allows the higher 
#grade without overfitting

for x in range(4,11):
    for y in range(4,8):
        df_merged_1 = df_merged[df_merged['Topic_Id'] < x].loc[df_merged['Update_Date'] > "2000-01-01"].loc[df_merged['Subject'] <y]
        Z = df_merged_1[["Subject","Topic_Id"]].to_numpy()
        # calculate the Silhouette score and BIC
        # for the number of clusters, k = 2 to 6
        S = []
        bic = []
        n_cluster_range = [2, 3, 4, 5]
        for n_cluster in n_cluster_range:
            gmm = mixture.GaussianMixture(n_components=n_cluster)
            gmm.fit(Z)
            lab = gmm.predict(Z)
            S.append(silhouette_score(Z, lab))
            bic.append(gmm.bic(Z))

        # show the resuls visuallay
        # figure with two plots
        fig, (ax1, ax2) = plt.subplots(1, 2)

        # first plot: Silhouette score
        ax1.plot(n_cluster_range, S)
        ax1.set_title('Silhouette Score')
        ax1.set(xlabel='Number of clusters', \
            ylabel='Silhouette Score')

        # second plot: BIC
        ax2.plot(n_cluster_range, bic)
        ax2.set_title('BIC')
        ax2.set(xlabel='Number of clusters', \
            ylabel='BIC')
        
        print(x,y)
        plt.show()

#%%GMM

Z = df_merged_1[["Subject","Topic_Id"]].to_numpy()

#%% specify Gaussian Mixture Model
gmm = mixture.GaussianMixture(n_components=3)

#%% fit the model
gmm.fit(Z)

#%%
# extract the clusters predictions according to
# the highest probability
labels = gmm.predict(Z)

#%% show results visually
plt.scatter(x=Z[:,0], y=Z[:,1], c=labels, cmap='viridis')
plt.show()
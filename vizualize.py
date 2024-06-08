import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data2 = pd.read_csv('reward_data_1.csv')
data1 = pd.read_csv('reward_data_1.csv')
data  = pd.read_csv('reward_data.csv')


data = data.append(data1)
data = data.append(data2)


clean_data = data.loc[(data['Reward']>0)&(data['Episode length']!=2000)]


clean_data['Time (second)'] = clean_data['Episode length']*(1./24.)
clean_data['Roughness (m)'] = clean_data['Roughness']

sns.set_theme(style="whitegrid", palette="bright")
sns.boxplot(x="Roughness (m)", y="Time (second)",hue="Terrain type",data=clean_data)
sns.despine(offset=10)
plt.show()
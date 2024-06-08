from tbparse import SummaryReader
import seaborn as sns
import matplotlib.pyplot as plt
log_dir = "events.out.tfevents.1709033054.32ca86f6f222.167.0"
reader = SummaryReader(log_dir)
df = reader.scalars
print(df['tag'].unique())
rew = df.loc[df['tag']=='train/critic_loss']
rew['Critic loss'] = rew['value']
rew['Step'] = rew['step']
print(rew)
sns.set_theme(style="whitegrid", palette="bright")
sns.lineplot(data=rew,x='Step',y='Critic loss',c='r')
sns.despine(offset=10)
plt.show()
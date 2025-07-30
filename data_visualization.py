import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('final2.csv')

def get_gender(row):
    if row['gender_male'] == 1:
        return 'Male'
    elif row['gender_female'] == 1:
        return 'Female'
    elif row['gender_non_binary'] == 1:
        return 'Non-binary'
    return 'Not specified'

df['gender'] = df.apply(get_gender, axis=1)

sns.set_style("whitegrid")

plt.figure(figsize=(10, 6))
sns.histplot(df['intention_score'], bins=10, kde=True)
plt.title('Distribution of Intention Score')
plt.xlabel('Intention Score')
plt.ylabel('Frequency')
plt.savefig('seaborn_intention_distribution.png')
plt.clf()

plt.figure(figsize=(10, 6))
sns.histplot(df['age'], bins=20, kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('seaborn_age_distribution.png')
plt.clf()

plt.figure(figsize=(8, 8))
sns.countplot(y=df['gender'])
plt.title('Distribution of Gender')
plt.xlabel('Count')
plt.ylabel('Gender')
plt.savefig('seaborn_gender_distribution.png')
plt.clf()


plt.figure(figsize=(10, 6))
sns.regplot(x='age', y='intention_score', data=df, scatter_kws={'alpha':0.3}, line_kws={"color": "red"})
plt.title('Intention Score vs. Age')
plt.xlabel('Age')
plt.ylabel('Intention Score')
plt.savefig('seaborn_intention_vs_age.png')
plt.clf()

plt.figure(figsize=(10, 6))
sns.boxplot(x='gender', y='intention_score', data=df)
plt.title('Intention Score by Gender')
plt.xlabel('Gender')
plt.ylabel('Intention Score')
plt.savefig('seaborn_intention_by_gender.png')
plt.clf()
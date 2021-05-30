import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../dataset/USA_Housing.csv')
print(df.head())

sns.pairplot(df)
plt.show()

# check data
print(df.isnull().sum())
# there is no null value

X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms',
        'Area Population']]
y = df['Price']

from sklearn.model_selection import train_test_split

print('Training...')
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, test_size=0.3)

from sklearn import linear_model

reg = linear_model.LinearRegression()

reg.fit(X_train, y_train)
print('Training done')

print(f'Training score: {reg.score(X_train, y_train)}')

print(f'Testing score: {reg.score(X_test, y_test)}')

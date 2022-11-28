import pandas
from sklearn import linear_model

df = pandas.read_csv("Confusion.csv")

X = df[['Score', 'Total Hint Views', 'Total Tip Views', 'Total Option Select', 'Attempt ']]
y = df['Confused']

regression = linear_model.LinearRegression()
regression.fit(X, y)
score = 6
totalHints = 6
totalTips = 23
totalOption = 13
attempt = 1

# Predict the confusion of the user where the Score is 6, Time taken is 306 secs and the Total Option Select is 13:
Confusion = regression.predict([[score, totalHints, totalTips, totalOption, attempt]])

print(abs(round(float(Confusion))))
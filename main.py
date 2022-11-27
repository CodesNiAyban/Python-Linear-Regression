import pandas
from sklearn import linear_model

df = pandas.read_csv("Confusion.csv")

X = df[['Score', 'Time Taken', 'Total Option Select']]
y = df['Confusion']

regr = linear_model.LinearRegression()
regr.fit(X, y)

# Predict the confusion of the user where the Score is 6, Time taken is 306 secs and the Total Option Select is 13:
predictedCO2 = regr.predict([[6, 306, 13]])

print(abs(round(float(predictedCO2))))
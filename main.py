import pandas
from sklearn import linear_model

df = pandas.read_csv("Confusion.csv")

X = df[['Score', 'Time Taken', 'Hint Click', 'Wrong Answer', 'Extra Option Click']]
y = df['Confused']

logReg = linear_model.LogisticRegression()  # Logistic Regression
logReg.fit(X.values, y.values)

Score = 8
TimeTaken = 321
HintClick = 4
WrongAnswer = 6
ExtraOptionClick = 8

# Predict the confusion of the user where the Score is 6, Time taken is 306 secs and the Total Option Select is 13:
Confusion = logReg.predict([[Score, TimeTaken, HintClick, WrongAnswer, ExtraOptionClick]])

print(Confusion)

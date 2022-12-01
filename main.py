# REGRESSION TRAINER AND FILE CREATOR
# import pandas
# from sklearn import linear_model
# import pickle
#
# df = pandas.read_csv("Confusion.csv")
#
# X = df[['Score', 'Hint Click', 'Wrong Answer', 'Extra Option Click']]
# y = df['Confused']
#
# logReg = linear_model.LogisticRegression()  # Logistic Regression
# logReg.fit(X.values, y.values)
#
# Score = 5
# TimeTaken = 321
# HintClick = 4
# WrongAnswer = 6
# ExtraOptionClick = 8
#
# # Predict the confusion of the user where the Score is 6, Time taken is 306 secs and the Total Option Select is 13:
# Confusion = logReg.predict([[Score, HintClick, WrongAnswer, ExtraOptionClick]])
# print(Confusion)
# pickle.dump(logReg, open('logReg.pkl', 'wb'))

# SERVER
from flask import Flask, request, jsonify
import numpy as np
import pickle

model = pickle.load(open('logReg.pkl', 'rb'))
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():

    score = np.array(request.form.get('Score'), dtype=float)
    hintClick = np.array(request.form.get('HintClick'), dtype=float)
    wrongAns = np.array(request.form.get('WrongAnswer'), dtype=float)
    optClick = np.array(request.form.get('ExtraOptionClick'), dtype=float)


    # Make prediction using model loaded from disk as per the data.
    input_query = np.array([[score, hintClick, wrongAns, optClick]])
    result = model.predict(input_query)[0]

    return jsonify({'placement': str(result)})


if __name__ == '__main__':
    app.run(port=5000, debug=True)

# Accuracy Test
# import pandas as pd
# from sklearn import metrics
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# import seaborn as sn
# df = pd.read_csv("Confusion.csv")
# X = df[['Score', 'Extra Option Click']]
# y = df['Confused']
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.25,random_state=False)
# logistic_regression = LogisticRegression()
# logistic_regression.fit(X_train,y_train)
# y_pred=logistic_regression.predict(X_test)
# print (X_test)
# print (y_pred)
# confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
# sn.heatmap(confusion_matrix, annot=True)
# accuracy = metrics.accuracy_score(y_test, y_pred)
# accuracy_percentage = 100 * accuracy
# print('Accuracy : ', accuracy)
# print("Accuracy Percentage (%) : ", accuracy_percentage)
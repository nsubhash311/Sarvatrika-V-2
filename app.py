from flask import Flask,render_template, url_for, request , redirect
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import json

app=Flask(__name__,template_folder='Sarvatrika',static_folder='Sarvatrika')

@app.route('/', methods=['GET','POST'])
def home():
    return render_template('home.html')

@app.route('/about')
def about():    
    return render_template('about.html')


@app.route('/her')
def her():    
    return render_template('answer.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/fetch', methods=['GET','POST'])
def fetch():
    if request.method == "POST":
        f = request.form['csvfile']
        with open(f) as file:
            global df
            df = pd.read_csv(file)
            col = [f"{c[0]}:{c[1]}" for c in enumerate(df.columns)]
            print(col)
            columns=[]
            print("loki")
            for i in col:
                columns.append(i[2:len(i)])
                print(i)
            print(columns)
            print(col)
            return render_template('variables.html',columns=columns)


@app.route('/answer', methods=['GET','POST']) 
def answer():
    if request.method == "POST":
        s = request.form['cValue']   
        dv = request.form['pridict']  
        print(s)
        print(type(s))
        iv=[]
        s=s.split(",")
        for i in s:
            iv.append(i.strip())
        iv.pop()
        print(iv)
        X = df[iv]
        dv = dv.strip()
        print(X)
        print(dv)
        y = df[dv]
        print(y)
        global X_train
        global X_test
        global y_train
        global y_test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        global flag
        global logisticModelScore
        print('see down')
        print(df[dv].isin([0,1]).all())
        if df[dv].isin([0,1]).all():
            flag = 1
            logisticModelScore = logisticRegression()
            linearModelScore = linearRegressor()
            decisionTreeModelScore = decisionTreeRegressor()
            lassoModelScore = lassoRegressor()
            bayesianRidgeModelScore = bayesianRidgeRegressor()
        else:
            flag = 0
            linearModelScore = linearRegressor()
            decisionTreeModelScore = decisionTreeRegressor()
            lassoModelScore = lassoRegressor()
            bayesianRidgeModelScore = bayesianRidgeRegressor()
            logisticModelScore = "Cannot be Applied !"
    return render_template('answer.html',linearModelScore=linearModelScore,decisionTreeModelScore=decisionTreeModelScore,lassoModelScore=lassoModelScore,bayesianRidgeModelScore=bayesianRidgeModelScore,logisticModelScore=logisticModelScore,len=len(iv),iv=iv,dv=dv)

def linearRegressor():
    global linearModel
    linearModel = linear_model.LinearRegression().fit(X_train, y_train)
    global linearModelScore
    linearModelScore = linearModel.score(X_test,y_test)
    linearModelScore*=100
    linearModelScore = round(linearModelScore,4)
    print("linearModelScore : ",linearModelScore)
    return linearModelScore

def decisionTreeRegressor():
    global decisionTreeModel
    decisionTreeModel = DecisionTreeRegressor().fit(X_train, y_train)
    global decisionTreeModelScore
    decisionTreeModelScore = decisionTreeModel.score(X_test,y_test)
    decisionTreeModelScore*=100
    decisionTreeModelScore = round(decisionTreeModelScore,4)
    print("decisionTreeModelScore : ",decisionTreeModelScore)
    return decisionTreeModelScore

def lassoRegressor():
    global lassoModel
    lassoModel = linear_model.Lasso(alpha=0.1).fit(X_train, y_train)
    global lassoModelScore
    lassoModelScore = lassoModel.score(X_test,y_test)
    lassoModelScore*=100
    lassoModelScore = round(lassoModelScore,4)
    print("lassoModelScore : " ,lassoModelScore)
    return lassoModelScore

def bayesianRidgeRegressor():
    global ridgeModel
    ridgeModel = linear_model.BayesianRidge().fit(X_train, y_train)
    global bayesianRidgeModelScore
    bayesianRidgeModelScore = ridgeModel.score(X_test,y_test)
    bayesianRidgeModelScore*=100
    bayesianRidgeModelScore = round(bayesianRidgeModelScore,4)
    print("bayesianRidgeModelScore : ",bayesianRidgeModelScore)
    return bayesianRidgeModelScore

def logisticRegression():
    global logisticModel
    logisticModel = LogisticRegression(solver='liblinear', random_state=0).fit(X_train, y_train)
    logisticModelScore = logisticModel.score(X_test,y_test)
    logisticModelScore*=100
    logisticModelScore = round(logisticModelScore,4)
    print("logisticModelScore : ",logisticModelScore)
    return logisticModelScore

@app.route('/predDataRes', methods=['GET','POST'])
def predDataRes():
    json_data = request.json
    arrColName = json_data["colName"]
    arrValue = json_data["data"]
    print(arrColName)
    print(arrValue)
    values = []
    for i in arrValue:
        values.append(float(i))
    print(values)
    if flag == 0:
        scores = [linearModelScore,decisionTreeModelScore,decisionTreeModelScore,bayesianRidgeModelScore]
    else:
        scores = [linearModelScore,decisionTreeModelScore,decisionTreeModelScore,bayesianRidgeModelScore,logisticModelScore]
    scores.sort(reverse=True)
    print(scores)
    if scores[0] == linearModelScore:
        res = linearModel.predict([values])
    elif scores[0] == decisionTreeModelScore:
        res = decisionTreeModel.predict([values])
    elif scores[0] == decisionTreeModelScore:
        res = lassoModel.predict([values])
    elif scores[0] == bayesianRidgeModelScore:
        res = ridgeModel.predict([values])
    else:
        res = logisticModel.predict([values])
    result = str(res[0])
    print(result)
    return result


if __name__ == "__main__":
    app.run()

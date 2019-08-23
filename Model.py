#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 23:07:55 2019

@author: nishithvn
"""
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt  
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from pyswip import Prolog
import pandas as pd
import numpy as np
import tkinter
import re
from sklearn.metrics import roc_auc_score


"""
Class to perform supervised learning for the given dataset.
User can configure whether or not to use the rules for a given trained model.
"""
class Model:
    
    def __init__(self, data, sensitive_feature, rule_option =None, new_confidence_level = None, rule= [], classifier_option=0):
        self.data = data
        self.sensitive_feature = sensitive_feature
        self.rule_option = rule_option
        self.new_confidence_level = new_confidence_level
        self.prolog = Prolog()
        self.rule = rule
        self.classifier_option = classifier_option
    
    def preprocess_features(self,X):
        '''
            Transform all categorical variables 
            and scale all features using scaler between zero and one.
        '''
        for column in X.columns:
            type_feature = "Numeric" if(bool(np.issubdtype(X[column].dtype, np.number))) else "Categorical"
            if (type_feature == "Categorical"):
                le = preprocessing.LabelEncoder()
                X[column].fillna(X[column].mode()[0], inplace=True)
                X[column] = le.fit_transform(X[column].astype(str))
            else:
                X[column].fillna(X[column].mode()[0], inplace=True)      
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)    
        return X
    
    def classify(self,test_data, use_rule=False):
        ''' 
          * both training and testing phase are performed here in this function
          * User can configure whether or not to use rules to postprocessing the predictions made by
          the classifier.
          * Logistic regression and Naive Bayes are the two supported probabilistic classifiers
          * Different model evaluation are computed before and after adding rules and displayed in a new frame
        '''

        columns = self.data.columns.tolist()
        output_column = columns[len(columns)-1]
        y_test = test_data[output_column]
        le = preprocessing.LabelEncoder()
        le.fit(self.data[output_column].unique())
        y_train = le.transform(self.data[output_column])
        y_test = le.transform(test_data[output_column])        
        X_train =  self.data.drop([output_column], axis=1)
        test=  test_data.drop([output_column], axis=1)
        X_train = self.preprocess_features(X_train.copy())
        X_test = self.preprocess_features(test.copy())
        test[self.sensitive_feature].fillna(test[self.sensitive_feature].mode()[0], inplace=True)
        classifier = self.getClassifier()
        classifier.fit(X_train, y_train)
        predicted_vals = classifier.predict(X_test)
        result = "The classification metrics after processing is given below \n"
        if (use_rule == False):
            y_pred = classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1score = f1_score(y_test, y_pred)
            result += "Accuracy of the model is "+ str(accuracy)+"\n"
            result += "F1 Score of the model is "+ str(f1score)+"\n"
        elif (self.rule_option==1):
            y_pred = classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1score = f1_score(y_test, y_pred)
            freq = self.data[self.sensitive_feature].value_counts()
            roc_auc = roc_auc_score(y_test, y_pred)
            result += "Without Logic Model (without postprocessing) \n"
            result += "Accuracy of the model is "+ str(accuracy)+"\n"
            result += "ROC AUC score"+ str(roc_auc)+"\n"
            result += "F1 Score of the model is "+ str(f1score)+"\n\n"
            se = pd.Series(y_pred)
            test['Predicted'] = se.values
            se = pd.Series(y_test)
            test['Actual'] = se.values
            previleged_group = freq.index[0]
            previleged_positive = len(test[(test[self.sensitive_feature]==previleged_group) & (test['Predicted']== 1)])
            privileged_total = len(test[(test[self.sensitive_feature]==previleged_group)])
            unprevileged_positive = len(test[(test[self.sensitive_feature]!=previleged_group) & (test['Predicted']== 1)])
            unprivileged_total = len(test[(test[self.sensitive_feature]!=previleged_group)])
            statistical_parity_difference = (previleged_positive/privileged_total)-(unprevileged_positive/unprivileged_total)
            result += "Privileged group "+str(previleged_group)+"\n"
            result += "Statistical parity difference is "+ str(statistical_parity_difference)+"\n\n"
            privileged_group = str.lower(str.strip(freq.index[0]))
            predicted_proba = classifier.predict_proba(X_test)
            new_y_pred = []
            self.build_logic_proportion_model(privileged_group)
            # if the user has not configured any rules
            if (not self.rule):
                count = 0
                for i,test_row in test.iterrows():
                    prob = predicted_proba[count][1]
                    new_y_pred.append(self.query_logic_proportion_model(test_row, prob))
                    count +=1
            else:
                final_rule = self.rule[-1]
                attribute_regex = r'(?<=\().*(?=\)\s:-)'
                attribute_vals = re.findall(attribute_regex,final_rule)
                attributes = attribute_vals[0].split(",")
                predicate_regex = r'[a-zA-Z]+(?=\(.*:-)'
                predicate = re.findall(predicate_regex,final_rule)[0]
                count = 0
                for i,test_row in test.iterrows():
                    query = predicate+'('
                    prob = predicted_proba[count][1]
                    #actual_val = predicted_vals[count]
                    count += 1
                    for attribute in attributes:
                        if (attribute=='Prob'):
                            query += str(prob)+","
                        elif (attribute=='P'):
                            query += 'P,'
                        else:
                            query += str.lower(str.strip(str(test_row[attribute])))+","
                    query = query.rstrip(",")+ ")"
                    qresult = list(self.prolog.query(query))
                    val = qresult[len(qresult)-1]['P']
                    new_y_pred.append(val)
            accuracy = accuracy_score(y_test, new_y_pred)
            f1score = f1_score(y_test, new_y_pred)
            roc_auc = roc_auc_score(y_test, new_y_pred)
            result += "With Logic Model (after postprocessing) \n"
            result += "Accuracy of the model is "+ str(accuracy)+"\n"
            result += "ROC AUC score"+ str(roc_auc)+"\n"
            result += "F1 Score of the model is "+ str(f1score)+"\n\n"
            se = pd.Series(new_y_pred)
            test['Label'] = se.values
            previleged_group = freq.index[0]
            privileged_test = test[(test[self.sensitive_feature]==previleged_group)]
            underprivileged_test = test[(test[self.sensitive_feature]!=previleged_group)]
            previleged_positive = len(test[(test[self.sensitive_feature]==previleged_group) & (test['Label']== 1)])
            privileged_total = len(test[(test[self.sensitive_feature]==previleged_group)])
            unprevileged_positive = len(test[(test[self.sensitive_feature]!=previleged_group) & (test['Label']== 1)])
            unprivileged_total = len(test[(test[self.sensitive_feature]!=previleged_group)])
            statistical_parity_difference = (previleged_positive/privileged_total)-(unprevileged_positive/unprivileged_total)
            result += "Statistical parity difference is "+ str(statistical_parity_difference)+"\n\n"
            privileged_y_actual = privileged_test['Actual']
            privileged_y_pred = privileged_test['Predicted']
            privileged_y_new = privileged_test['Label']
            result += "Before postprocessing for Privileged group- "+str(privileged_group)+"\n"
            accuracy = accuracy_score(privileged_y_actual, privileged_y_pred)
            f1score = f1_score(privileged_y_actual, privileged_y_pred)
            roc_auc = roc_auc_score(privileged_y_actual, privileged_y_pred)
            result += "Accuracy of the model is "+ str(accuracy)+"\n"
            result += "ROC AUC score"+ str(roc_auc)+"\n"
            result += "F1 Score of the model is "+ str(f1score)+"\n\n"
            result += "After postprocessing for Privileged group- "+str(privileged_group)+"\n"
            accuracy = accuracy_score(privileged_y_actual, privileged_y_new)
            f1score = f1_score(privileged_y_actual, privileged_y_new)
            roc_auc = roc_auc_score(privileged_y_actual, privileged_y_new)
            result += "Accuracy of the model is "+ str(accuracy)+"\n"
            result += "ROC AUC score"+ str(roc_auc)+"\n"
            result += "F1 Score of the model is "+ str(f1score)+"\n\n"
            unprivileged_y_actual = underprivileged_test['Actual']
            unprivileged_y_pred = underprivileged_test['Predicted']
            unprivileged_y_new = underprivileged_test['Label']
            result += "Before postprocessing for Underprivileged group- Non-"+str(privileged_group)+"\n"
            accuracy = accuracy_score(unprivileged_y_actual, unprivileged_y_pred)
            f1score = f1_score(unprivileged_y_actual, unprivileged_y_pred)
            roc_auc = roc_auc_score(unprivileged_y_actual, unprivileged_y_pred)
            result += "Accuracy of the model is "+ str(accuracy)+"\n"
            result += "ROC AUC score"+ str(roc_auc)+"\n"
            result += "F1 Score of the model is "+ str(f1score)+"\n\n"
            result += "After postprocessing for Underprivileged group- Non-"+str(privileged_group)+"\n"
            accuracy = accuracy_score(unprivileged_y_actual, unprivileged_y_new)
            f1score = f1_score(unprivileged_y_actual, unprivileged_y_new)
            roc_auc = roc_auc_score(unprivileged_y_actual, unprivileged_y_new)
            result += "Accuracy of the model is "+ str(accuracy)+"\n"
            result += "ROC AUC score"+ str(roc_auc)+"\n"
            result += "F1 Score of the model is "+ str(f1score)+"\n\n"
        self.showOutput(result)
        return accuracy

    def getClassifier(self):
        """
        Get classifie based on the user selection
        """
        if(self.classifier_option == "Logistic Regression"):
            return LogisticRegression(random_state=0)
        else:
            return GaussianNB()

    def showOutput(self,result):
        """
        All the evaluation metrics computed by the model are displayed in a frame
        using this function
        """
        window = tkinter.Tk()
        window.wm_title("Output")
        frame = tkinter.Frame(window, bd=1, relief="solid")
        op = tkinter.Text(frame, wrap="word", font="Arial")
        op.pack(fill="x")
        op.insert("end",result)
        frame.pack(fill="x", padx=20, pady=10) 
    
    def query_logic_proportion_model(self, test_row, prob):
        """ 
        the rules that are configured by the user are queried upon for each
        instance of the test row to get the prediction obtained by the logic model
        """
        test_sensitive_feature = str.lower(str.strip(test_row[self.sensitive_feature]))        
        logic_query = "fairModel("+str(test_sensitive_feature)+","+str(prob)+",P)"
        result = list(self.prolog.query(logic_query))
        return result[len(result)-1]['P']
            
        
    def build_logic_proportion_model(self,privileged_group):
        """
        the user can configure their own rules or the framework generates a custom rule
        based on the proportion of srnsitive groups in the trained data
        """
        if (not self.rule):
            self.prolog.assertz("isNotPrivileged(X):- X \= '"+privileged_group+"'")
            self.prolog.assertz("overrideProbability(M,N):- M > N")
            self.prolog.assertz("fairModel(X,Prob,Z) :- isNotPrivileged(X) -> (overrideProbability(Prob,"+str(self.new_confidence_level)+") ->  Z is 1 ; Z is 0); overrideProbability(Prob,0.5) ->  Z is 1 ; Z is 0")
        else:
            for custom_rule in self.rule:
                self.prolog.assertz(custom_rule)
                

        
        
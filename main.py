#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 22:27:25 2019

@author: nishithvn
"""

import tkinter
from tkinter import filedialog, font,simpledialog
from tkinter import messagebox
from tkinter.messagebox import showerror
import ntpath
import pandas as pd
import numpy as np
import csv
from PIL import Image
from tables import createStandardTable as cst
from sklearn import preprocessing
from plotnine import *
import matplotlib.pyplot as plt
from Model import Model
import time

sec_button_pressed = False

def import_data():
    """
    this reads the data from the csv file and displays the desciptive stats of the training data
    sets up all the UI components in the initial window
    """
    global training_data
    opts = {'filetypes': [('text files', '.csv')]}
    filepath = filedialog.askopenfilename(**opts)
    filename = ntpath.basename(filepath)
    if filename != "":
        messagebox.showinfo("File Imported", "Choose an appropriate action you want to perform")
        data = pd.read_csv(filepath)
        file_data = data
        column_mappings = {"Feature": [], "Type": [], "Missing": [],"Total": [], "Max" :[], "Min" :[], "Mean":[], "SD":[]}
        columns = data.columns.tolist()
        for column in columns:
            column_mappings["Feature"].append(column)
            column_mappings["Missing"].append(len(data[data[column].isnull()]))
            type_feature = "Numeric" if(bool(np.issubdtype(data[column].dtype, np.number))) else "String"
            if (len(data[column].unique()) ==2) :
                type_feature += " / Binary"
            if (type_feature == "Numeric"):
                column_mappings["Max"].append(data[column].max())
                column_mappings["Min"].append(data[column].min())
                column_mappings["Mean"].append(round(data[column].mean(),2))
                column_mappings["SD"].append(round(data.loc[:,column].std(),2))
            else:
                column_mappings["Max"].append("NA")
                column_mappings["Min"].append("NA")
                column_mappings["Mean"].append("NA")              
                column_mappings["SD"].append("NA")              
            column_mappings["Type"].append(type_feature)
            column_mappings["Total"].append(len(data))
        data = pd.DataFrame(column_mappings)
        data.to_csv('imported_data.csv', index=False)
        create_table_frame()
        OPTIONS = columns
        OPTIONS = OPTIONS[:-1]
        variable.set(OPTIONS[0])
        w = tkinter.OptionMenu(root_win, variable, *OPTIONS)
        output_column = columns[len(columns)-1]
        y = file_data[output_column]        
        training_data = file_data.drop([output_column], axis=1)
        le = preprocessing.LabelEncoder()
        le.fit(file_data[output_column].unique())
        y = le.transform(file_data[output_column])
        se = pd.Series(y)
        training_data['Label'] = se.values
        labeltext = tkinter.StringVar()
        label = tkinter.Label(root_win, textvariable=labeltext, font=("Arial",16 ), relief="groove")
        labeltext.set("Dataset Features")
        label.place(x=30, y=160)
        add_sec_button = tkinter.Button(root_win, text=("Add Secondary Feature"),borderwidth=2,command=lambda: add_secondary_features(file_data))
        add_sec_button.place(x= 350, y=160)
        w.place(x=230, y=160)
        classifier_options = ["Logistic Regression","Naive Bayes"]
        classifier_variable.set(classifier_options[0])
        classifier_opt = tkinter.OptionMenu(root_win, classifier_variable, *classifier_options)
        classifier_opt.place(relx=0.55, rely=.57, anchor="se")
        y = 300
        v.set(0)
        for val, opti in enumerate(radio_options):
            tkinter.Radiobutton(root_win, font=font.Font(family='Times', size=15, weight=font.BOLD),
                                text=opti,variable=v, pady=5,value=val).place(x=200, y=y)
            y += 70
        process_button = tkinter.Button(root_win, text=("Process"),borderwidth=2,command=lambda: get_option(file_data))
        process_button.place(relx=0.9, rely=.75, anchor="se")
        rule_button = tkinter.Button(root_win, text=("Upload Rule"),borderwidth=2,command=upload_rule)
        rule_button.place(relx=0.9, rely=.6, anchor="se")
    
def upload_rule():
    """
    this lets user to upload the custom prolog rules
    """
    global rules
    opts = {'filetypes': [('text files', '.pl')]}
    filepath = filedialog.askopenfilename(**opts)
    filename = ntpath.basename(filepath)
    if filename != "":
        messagebox.showinfo("File Imported", "Rules have been successfully imported")
        f1 = open(filepath)
        lines = f1.readlines()
        del rules[:] 
        for line in lines:    
            rules.append(line.rstrip('.\n'))

def add_secondary_features(data):
    """
    lets user to add secondary feature based on which they can visualize and check fairness
    """
    global sec_button_pressed 
    OPTIONS = data.columns.tolist()
    OPTIONS = OPTIONS[:-1]
    sec_variable.set(OPTIONS[0])
    w = tkinter.OptionMenu(root_win, sec_variable, *OPTIONS)
    w.place(x=550, y=160)
    sec_button_pressed = True


def show_pie_distribution(attribute, data):
    """
    shows pie distribution of the sensitive attributes selected by the user
    """
    inp_data = data.copy()
    if(sec_button_pressed):
        sec_attribute = sec_variable.get()
        inp_data[attribute] = inp_data[[attribute, sec_attribute]].apply(lambda x: '-'.join(x), axis=1)
    x = inp_data[attribute].value_counts().plot(kind='pie')
    plt.axis('equal')
    x.figure.savefig('pie-plot.png')
    img = Image.open('pie-plot.png')
    img.show()
    plt.clf()



def get_option(data):
    attribute = variable.get()
    chosen_option = v.get()
    if (chosen_option ==0):
        show_histogram(attribute, data)
    elif (chosen_option ==1):
        show_pie_distribution(attribute, data)
    elif (chosen_option ==2):
        show_label_distribution(attribute, data)
    elif (chosen_option ==3):
        messagebox.showinfo("Import Test data", "Import Test Data to Predict")
        opts = {'filetypes': [('text files', '.csv')]}
        filepath = filedialog.askopenfilename(**opts)
        if filepath != "":        
            test_data = pd.read_csv(filepath)
            build_model(attribute,data, test_data)
    elif (chosen_option ==4):
        compute_statistical_parity_difference(attribute, data)

def compute_statistical_parity_difference(attribute, data):
    """
       Statistical Parity Difference = Pr(Y = 1 | X = privileged) - Pr(Y = 1 | X ≃ privileged)

    """
    freq = data[attribute].value_counts()
    previleged_group = freq.index[0]
    previleged_positive = len(training_data[(training_data[attribute]==previleged_group) & (training_data['Label']== 1)])
    privileged_total = len(training_data[(training_data[attribute]==previleged_group)])
    unprevileged_positive = len(training_data[(training_data[attribute]!=previleged_group) & (training_data['Label']== 1)])
    unprivileged_total = len(training_data[(training_data[attribute]!=previleged_group)])
    statistical_parity_difference = (previleged_positive/privileged_total)-(unprevileged_positive/unprivileged_total)
    if (statistical_parity_difference > 0.05):
        showerror("Statistical Parity Difference", abs(statistical_parity_difference))
    else:
        messagebox.showinfo("Statistical Parity Difference", abs(statistical_parity_difference))

def build_model(attribute, data, test_data):
    type_feature = "Numeric" if(bool(np.issubdtype(data[attribute].dtype, np.number))) else "Categorical"
    classifier_option = classifier_variable.get()
    if (type_feature == "Categorical"):
        if (len(rules)>0):
            model =  Model(data,attribute,1,rule=rules, classifier_option=classifier_option)
            model.classify(test_data, True)
        else:
            freq = data[attribute].value_counts()
            freq_lis = freq.tolist()
            max_elem = freq_lis.pop(0)
            previleged_group = freq.index[0]
            if (sum(freq_lis)< max_elem*0.25) :
                show_histogram(attribute, data)
                messagebox.showinfo("Inappropriate Distribution", "From the frequency plot,the proportion of privileged (high proportion) group "+str(previleged_group) + " is more than 4 times the proportion of Non "+str(previleged_group)+ " group")
                ans = simpledialog.askfloat('Update confidence level (0<x<1)', 'Please provide the new confidence level for Non '+str(previleged_group)+ " group")
                model =  Model(data,attribute,1,ans,rule=rules,classifier_option=classifier_option)
                model.classify(test_data,True)
            else:
                model =  Model(data,attribute, classifier_option=classifier_option)
                model.classify(test_data, False)    
    else:
        model =  Model(data,attribute, classifier_option=classifier_option)
        model.classify(test_data,False)
                

def show_label_distribution(attribute, data):
    """
    shows label distribution of the sensitive attributes selected by the user
    """
    columns = data.columns.tolist()
    data[attribute].fillna(data[attribute].mode()[0], inplace=True)  
    p = ggplot(data,aes(x=attribute)) + geom_bar(width=0.1) +  facet_wrap(columns[len(columns)-1]) + theme_gray() +theme_matplotlib(rc={"figure.figsize": "15, 15"})
    t = theme_gray()
    t._rcParams['font.size'] = 30 # Legend font size
    t._rcParams['xtick.labelsize'] = 11 #
    t._rcParams['ytick.labelsize'] = 11 #
    #use this to change the size of the visualizations
    #t._rcParams["figure.figsize"] = "13, 13"
    p = p + t
    p.save('plot2.png')
    img = Image.open('plot2.png')
    img.show()
    plt.clf()
    
def show_histogram(attribute, data):
    """
    shows histogram distribution of the sensitive attributes selected by the user
    """
    type_feature = "Numeric" if(bool(np.issubdtype(data[attribute].dtype, np.number))) else "String"
    if (type_feature == "String"):
        if(sec_button_pressed):
            sec_attribute = sec_variable.get()
            data[attribute] = data[[attribute, sec_attribute]].apply(lambda x: '-'.join(x), axis=1)
        x = data[attribute].value_counts().plot(kind='bar')
        #x.set_xticklabels(data[attribute].unique(), rotation=45)
        x.figure.savefig('plot.png')
    else:
        fig, ax = plt.subplots()
        data.hist(attribute, ax=ax)
        fig.savefig('plot.png')
    img = Image.open('plot.png')
    img.show()
    plt.clf()
    
def hide_table_frame(tableFrame,window):
    """
    hides descriptive stats shown after importing training data
    """    
    tableFrame.grid_remove()
    tableFrame.destroy()
    window.destroy()



def create_table_frame():
    """
    the descriptive statistics are displayed in a new frame 
    by reading the stats stored in imported_data.csv
    [imported_data.csv is stored after importing training data]
    """
    f = open("imported_data.csv")
    window = tkinter.Tk()
    window.wm_title("Feature Statistics")
    #window.geometry("500x500")
    tableFrame = tkinter.Frame(window)
    newtable = cst(f,tableFrame)
    newtable.grid()
    tkinter.Button(tableFrame,text="Close",command=lambda: hide_table_frame(tableFrame,window)).grid()
    tableFrame.grid()

# the different options that the user can select from after importing the training data
radio_options = [
        "Histogram Plot",
        "Pie Plot", 
        "Label Distribution",
        "Classifier",
        "Statistical Parity Difference"
]

## Setting up the GUI window params
root_win = tkinter.Tk()
root_win.wm_title("SRL")
root_win.wm_attributes('-alpha', 1.00)
fig = plt.figure(frameon = True)
fig.set_size_inches(17, 15)
# Use this setup the icon for the application
#root_win.wm_iconbitmap(bitmap = "@icon.xbm")
helv14 = font.Font(family="Helvetica",size=16,weight="bold")
# change the dimensions of the window based on the requirements
root_win.geometry("950x950")
file_import_text = tkinter.StringVar()
import_button = tkinter.Button(root_win,  height=2, width=17,borderwidth=2,font=helv14, textvariable= file_import_text,command=import_data)
file_import_text.set('Upload CSV file')
import_button.place(relx=.9, rely=.5, anchor="c")
label_text = tkinter.StringVar()
label = tkinter.Label(root_win, textvariable=label_text, font=helv14,bg="white", relief="groove")
label_text.set("FairSRL")
#initialize parameters
rules = []
label.place(relx=.8, rely=.2, anchor="c")
classification_details_container = tkinter.Frame()
classification_details_label = tkinter.Label()
display_output = False
training_data = None
v = tkinter.IntVar()
variable = tkinter.StringVar(root_win)
sec_variable = tkinter.StringVar(root_win)
classifier_variable = tkinter.StringVar(root_win)
root_win.mainloop()
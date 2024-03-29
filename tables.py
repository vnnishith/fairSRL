from tkinter import *
import csv


"""
Source: https://github.com/clarenceangel/tkinterstuff
Used to create a new frame from a csv file to display all the descriptive 
statistics of the dataset
"""
def createStandardTable(f,window):
    handle = csv.reader(f)
    length = len(next(handle))

    sizes = [0] * length
    for record in handle:
        
        for p,column in enumerate(record):
            if len(column) > sizes[p]:
                sizes[p] = len(column)+5
                
    f.seek(0)
    trow = 0
    table = Frame(window)
    for record in handle:
        for w,column in enumerate(record):
            Label(table,text=column,width=sizes[w],borderwidth=2,relief="groove",justify=LEFT,anchor=W, background='white').grid(column=w,row=trow,sticky=W)
        
        trow+=1
        
    return table

    


        
            

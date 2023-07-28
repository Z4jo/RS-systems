import random as rnd
import numpy as np
import pandas as pd
import re
with open('./cross_validation/results.csv','r') as file:
    f = file 
    
    rows = 9
    cols = 7
    mean = [[0 for _ in range(cols)] for _ in range(rows)]   
    name = ""
    start = -1
    for line in f:
        splited_line = line.split(",")

        splited_name = re.split(r'_|\.',splited_line[0],flags=re.MULTILINE | re.DOTALL)
        print(splited_name)
        number_of_iteration = splited_name[len(splited_name)-2]

        if len(number_of_iteration) > 1 or int(number_of_iteration) == 0:
            continue
        if splited_name[0] != name:
            start += 1
            name = splited_name[0]
        for i in range(1,len(splited_line)):
            mean[start][i-1]+=float(splited_line[i])
            #print(mean[start][i-1])

    #print(mean[0])
    #print(mean[1])
    #print(mean[2])
    #print(mean[3])
    #print(mean[4])
    #print(mean[5])
    #print(mean[6])
    #print(mean[7])
    #print(mean[8])
    #print(mean[0][0])
    #print("-------------------")
     
    for i in range(cols):
        #print(f"0;{i} mean:{mean[0][i]}, mean/4: {mean[0][i]/4}")
        mean[0][i] = mean[0][i]/4
        #print(f"1;{i} mean:{mean[1][i]}, mean/4: {mean[1][i]/4}")
        mean[1][i] = mean[1][i]/4
        #print(f"2;{i} mean:{mean[2][i]}, mean/4: {mean[2][i]/4}")
        mean[2][i]= mean[2][i]/4
        #print(f"3;{i} mean:{mean[3][i]}, mean/4: {mean[3][i]/4}")
        mean[3][i]= mean[3][i]/4
        #print(f"4;{i} mean:{mean[4][i]}, mean/4: {mean[4][i]/4}")
        mean[4][i]= mean[4][i]/4
        #print(f"5;{i} mean:{mean[5][i]}, mean/4: {mean[5][i]/4}")
        mean[5][i]= mean[5][i]/4
       # print(f"6;{i} mean:{mean[6][i]}, mean/4: {mean[6][i]/4}")
        mean[6][i]= mean[6][i]/4
       # print(f"7;{i} mean:{mean[7][i]}, mean/4: {mean[7][i]/4}")
        mean[7][i]= mean[7][i]/4
       # print(f"8;{i} mean:{mean[8][i]}, mean/4: {mean[8][i]/4}")
        mean[8][i]= mean[8][i]/4

    print(mean[0])
    print(mean[1])
    print(mean[2])
    print(mean[3])
    print(mean[4])
    print(mean[5])
    print(mean[6])
    print(mean[7])
    print(mean[8])
        


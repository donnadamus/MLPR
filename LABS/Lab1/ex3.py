#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: marcodonnarumma
"""

citydict = {}
monthdict = {}

monthnumber = {1: 'Jan', 2: 'Feb', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
               7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November',
               12: 'December'}

f = open("ex3_data.txt", "r")

for line in f:
    fields = line.split(" ")
    date = fields[3].split("/")
    
    city = fields[2]
    month = int(date[1])
    
    citydict[city] =  citydict[city] + 1 if city in citydict else 1
    monthdict[month] =  monthdict[month] + 1 if month in monthdict else 1
    
print("Births per city: ")

for k in citydict:
    print(k, citydict[k])
    
    
print("\n\nBirths per month: ")

for k in monthdict:
    print(monthnumber[k], monthdict[k])
    
print("\n\nAverage number of births: ", sum(citydict.values())/len(citydict.keys()))
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: marcodonnarumma
"""

class Competitor:
    def __init__(self, name, surname, country, scores):
        self.name = name
        self.surname = surname
        self.country = country
        self.scores = scores
    def __str__(self):
        return self.name + " " + self.surname + " " + self.country + " " + str(self.scores)
        


f = open("ex1_data.txt", "r")
lines = []

competitors = []

for line in f:
    fields = line.strip().split(" ")
    scores = sorted([float(val) for val in fields[3:]])
    scores.pop(0)
    scores.pop() # default remove last
    sumscores = sum(scores)
    competitors.append(Competitor(fields[0], fields[1], fields[2], sumscores))
    

competitors.sort(key=lambda x: x.scores, reverse = True)

for i in range(0, 3):
    print(competitors[i])
    
dizionario = {}

for c in competitors:
    if c.country in dizionario:
        dizionario[c.country] += c.scores
    else:
        dizionario[c.country] = c.scores

    
        
maxcountry = max(dizionario, key=dizionario.get)

print(maxcountry + " " + str(dizionario[maxcountry]))
    
    
    
    
    
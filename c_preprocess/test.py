# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 11:11:15 2019

@author: bzli
"""

def test(a, b):
    print(id(a))
    a.extend(b)
    print(id(a))
    a = set(a)
    print(id(a))
    
x=[1,2,3]
y=[1,2,3]
test(x, y)
print(x)

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 18:45:54 2019

@author: bzli
"""
import os
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
from Model_batch import Model
import random
import os
from GraphStructure import Graph
import GraphStructure
import pickle
import shutil

countMap = {}
def loadData():
    dirs = os.listdir(dataFilePath)
    for classPathName, classIndex in zip(dirs, range(len(dirs))):
        for graphPathName in os.listdir(dataFilePath + classPathName):
            graph = Graph(dataFilePath + classPathName + "/" + graphPathName, priorList, "#")
            graphList.append(graph)
            labelList.append(classIndex)
            
def deleteData():
    dirs = os.listdir(dataFilePath)
    for classPathName, classIndex in zip(dirs, range(len(dirs))):
        #删除样本少于三个的类
        if len(os.listdir(dataFilePath + classPathName)) < 3:
            shutil.rmtree(dataFilePath + classPathName)
            continue
        
        
        for graphPathName in os.listdir(dataFilePath + classPathName):
            #删除存在多个函数的样本
            if len(os.listdir(dataFilePath + classPathName + "/" + graphPathName)) > 3 or len(os.listdir(dataFilePath + classPathName + "/" + graphPathName)) == 0:
                shutil.rmtree(dataFilePath + classPathName + "/" + graphPathName)
        #删除过后如果该类别目录空了则移除该目录
        if len(os.listdir(dataFilePath + classPathName)) == 0:
            shutil.rmtree(dataFilePath + classPathName)
            continue
        
def deleteData2():
    dirs = os.listdir(dataFilePath)
    for classPathName, classIndex in zip(dirs, range(len(dirs))):
        for graphPathName in os.listdir(dataFilePath + classPathName):
            f = open(dataFilePath + classPathName + "/" + graphPathName+"/"+"main.txt")
            con = f.read()
            f.close()
            if con.find("unknown")>=0:
                print(dataFilePath + classPathName + "/" + graphPathName+"/"+"main.txt")
                shutil.rmtree(dataFilePath + classPathName+ "/" + graphPathName)

dataFilePath = "E:/program_embedding_literature/OJ_CLONE_217/"
priorList = {'assign':-5,"return":-5,"param":-2,"~":6 ,'p++':6,"p--":6,"+":4, "-":4, "++":6,"--":6,"*":5 ,"/":5, "%":5, "!=":1, "==":1,
             "<":2,">":2,"<=":2,">=":2,"&&":-2,"&":0,"||":-3,"|": -1,"c_*":6, "c_+":6,"c_-":6,"^":6,
             "invoke":-4, "parammix":-3, "sizeof":6 ,"<<":3, ">>":3,"!":6, "c_&":6,"c_|":6, "(":7,")":7,"structureaccess":7}

graphList = []
labelList = []

#deleteData()
#deleteData2()
GraphStructure.setTopK(20)
loadData()
nums = []
maxNode = 0
for graph in graphList:
    adjMat,nodes,_=graph.getGraphInfo()
    if len(nodes) > maxNode:
        maxNode = len(nodes)
    try:
        graph.getMetaData()[0].extend(graph.getMetaData()[1])
    except Exception as e:
        shutil.rmtree(graph.getMethodFilePath())
        continue
    nums.append(len(graph.getMetaData()[0]))
print("samples:", len(nums))
nums = np.array(nums)
print("max nodes:", maxNode)
print("mean:", nums.mean())
print("min:", nums.min())
print("max:", nums.max())
print("std:", nums.std())
print("finish")
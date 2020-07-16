# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 14:10:33 2019

@author: bzli
"""
import os
import numpy as np
import copy

def getNodeEdgeNum(filePath):
    f = open(filePath, "r")
    lines = f.readlines()
    nodeCount = 0
    edgeCount = 0
    for line in lines:
        if line.split('\t')[0] == "node":
            nowIndex = int(line.split('\t')[1].replace("node",""))
            if nowIndex > nodeCount:
                nodeCount = nowIndex
                continue
        if line.split('\t')[0] == "edge":
            nowIndex = int(line.split('\t')[1].replace("edge",""))
            if nowIndex > edgeCount:
                edgeCount = nowIndex
                continue
    return nodeCount+1, edgeCount

class Graph:
    def __init__(self, methodFilesPath, priorList, delima):
        self.priorList = priorList
        self.delima = delima
        self.subGraphs = []
        self.methodFilesPath = methodFilesPath
        for fileName in os.listdir(methodFilesPath):
            strs = fileName.split(".")
            if strs[-1] == "data":
                self.__dealMetaInfo__(os.path.join(methodFilesPath, fileName))
                continue
            if strs[-1]=="txt":
                self._dealMethodInfo__(os.path.join(methodFilesPath, fileName))
        self.mergeAdjMat = None    
        self.className = None
        
    def __dealMetaInfo__(self, filePath):
        f = open(filePath, 'r')
        lines = f.readlines()
        #处理变量列表
        variableList = [ele.split(':') for ele in lines[0].split('\t')][0:-1]
        variableList = [variable for variable in variableList if variable[0] != ""]
        #print(variableList)
        sorted(variableList, key = lambda x:int(x[1]))
        self.variableList = variableList                                                              
        #处理方法列表
        methodList = [ele.split(":") for ele in lines[1].split('\t')][0:-1]
        methodList = [method for method in methodList if method[0] != ""]
        #print(methodList)
        sorted(methodList, key = lambda x:int(x[1]))
        self.methodList = methodList
    def getMethodFilePath(self):
        return self.methodFilesPath
    
    def getClassName(self):
        return self.getMethodFilePath().split("/")[-2]

    
    def getMetaData(self):
        #print(self.methodFilesPath)
        return self.variableList, self.methodList
    
    def getSubGraphs(self):
        return self.subGraphs
    def __newAdjMat__(self, adjMat):
        for i in range(len(adjMat)):
            for j in range(len(adjMat[i])):
                if adjMat[i][j] != 0:
                    adjMat[j][i] = adjMat[i][j]
        return adjMat
    
    def getGraphInfo(self):
        #print(self.methodFilesPath)
        if self.mergeAdjMat is not None:
            return self.mergeAdjMat, self.mergeNodes, self.mergeEdges
#        try:
        adjMat, nodes, edges = self.subGraphs[0].getAdjMat(), self.subGraphs[0].getNodes(), self.subGraphs[0].getEdges()
#        except Exception as e:
#            print(e)
        #print(self.subGraphs[0].getFilePath())
#        for structure2 in self.subGraphs[1:]:
#            #print(structure2.getFilePath())
#            adjMat2, nodes2, edges2 = structure2.getAdjMat(), structure2.getNodes(), structure2.getEdges()
#            adjMat, nodes, edges = self.__mergeStructure__([adjMat, adjMat2], [nodes, nodes2], [edges, edges2])
        countMap = {}
        for node in nodes:
            chainList = node["chainList"]
            for ele in chainList:
                if ele['type'].lower().find("entity") >= 0:
                    if countMap.get(ele["type"]) == None:
                        countMap[ele["content"]] = 1
                    else:
                        countMap[ele["content"]] += 1
        l = list(countMap.items())
        l.sort(key = lambda x:x[1])
        topKEle = [i[0] for i in l]
        for node in nodes:
            chainList = node["chainList"]
            for ele in chainList:
                if ele['type'].lower().find("entity") >= 0:
                    index = topKEle.index(ele["content"])
                    if index < TopK:
                        ele["content"] = "top_"+str(index)
                    else:
                        ele["content"] = "top_other"

            
        self.mergeAdjMat = self.__newAdjMat__(adjMat)
        self.mergeNodes = nodes
        self.mergeEdges = edges
        #print(self.mergeAdjMat.shape)
#        for i in range(self.mergeAdjMat.shape[0]):
#            if np.sum(self.mergeAdjMat[i,:])==0 and np.sum(self.mergeAdjMat[:,i]==0):
#                print(self.mergeAdjMat)
#                print(i)
#                raise Exception()
        return adjMat, nodes, edges
    
    def __mergeStructure__(self, mats, nodes, edges):
        #print(self.methodFilesPath)
        s1Mat = copy.deepcopy(mats[0])
        s2Mat = copy.deepcopy(mats[1])
        s1Nodes = copy.deepcopy(nodes[0])
        s2Nodes = copy.deepcopy(nodes[1])
        s1Edge = copy.deepcopy(edges[0])
        s2Edge = copy.deepcopy(edges[1])
        arrayShape = [s1Mat.shape[0]+s2Mat.shape[0]-1, s1Mat.shape[1]+s2Mat.shape[1]-1]
        mergedArray = np.array(np.zeros(arrayShape))
        
        #合并邻接矩阵
        mergedArray[0:s1Mat.shape[0], 0:s1Mat.shape[1]] = s1Mat
        mergedArray[s1Mat.shape[0]:, s1Mat.shape[1]:] = s2Mat[1:,1:]
        
        #合并Node
        s1Nodes.extend(s2Nodes[1:])
        
        #合并Edge
        edge = {"label":"startEdge"}
        s1Edge.extend([edge])
        newEdgeIndex = len(s1Edge)-1
        s1Edge.extend(s2Edge[2:])
        
        #修正邻接矩阵的边index
        tempMat = s2Mat[1:,1:]
        whereTuple = np.where(tempMat>0)
        if len(whereTuple[0]) == 0:
            mat2Min = 0
        else:
            mat2Min = tempMat[np.where(tempMat>0)].min()
        mergedArray[s1Mat.shape[0]:, s1Mat.shape[1]:][np.where(mergedArray[s1Mat.shape[0]:, s1Mat.shape[1]:]>0)] += (newEdgeIndex - int(mat2Min) + 1)
        #修正新邻接矩阵的起始边
#        try:
        mergedArray[0, s1Mat.shape[1]] = newEdgeIndex
#        except Exception as e:
#            print(e)
        
        return  mergedArray, s1Nodes, s1Edge
    
    def _dealMethodInfo__(self, filePath):
        nodeNum, edgeNum = getNodeEdgeNum(filePath)
        gs = GraphStructure(filePath, nodeNum, edgeNum,self.delima, self.priorList)
        self.subGraphs.append(gs)
    
class GraphStructure:
    def __init__(self, graphFilePath, maxNodeNum, maxEdgeNum, dilema, priorList):
        graphFile = open(graphFilePath, 'r')
        self.gfp = graphFilePath
        self.lines = graphFile.readlines()
        self.maxNodeNum, self.maxEdgeNum = getNodeEdgeNum(graphFilePath)
        self.dilema = dilema
        self.priorList = priorList
        self.adjMatrix = np.zeros([maxNodeNum, maxNodeNum])
        self.__intitDataStructure__()
        self.__construcLine__()
        self.__clearInvalidNode__()
    
    def __intitDataStructure__(self):
        self.nodes = [None] * (self.maxNodeNum)
        self.edges = [None] * (self.maxEdgeNum+1)
        
    #分发到每行数据相应类型的处理函数    
    def __construcLine__(self):
        for line in self.lines:
            elementType = line.split("\t")[0]
            if elementType == "node":
                self.__constructNode__(line.strip())
                continue
            if elementType == "edge":
                self.__constructEdege__(line.strip())
                continue
    
    #处理edge类型的行
    def __constructEdege__(self,line):
        temp = line.split("\t")
        edgeIndex = int(temp[1].replace("edge",""))
        nodes = temp[2].split('->')
        headIndex = int(nodes[0].replace("node",""))
        tailIndex = int(nodes[1].replace("node",""))
        self.adjMatrix[headIndex, tailIndex] = edgeIndex
        if len(temp)<4:
            edgeInfo = {"label":";"}
        else:
            edgeInfo = {"label":temp[3]}
        self.edges[edgeIndex] = edgeInfo
        return edgeInfo
    
    #处理node类型的行
    def __constructNode__(self, line):
        temp = line.split("\t")
        #print(self.gfp)
        #print(temp)
        nodeIndex = int(temp[1].replace("node",""))
        nodeType = temp[2]
        nodeContent = temp[3]
        invokeList = self.__dealInvokeChain__(nodeContent)
        nodeInfo = {"nodeType":nodeType,"content":nodeContent.strip(), "chainList":invokeList}
        self.nodes[nodeIndex] = nodeInfo
    
    def __clearInvalidNode__(self):
        while len([index for n,index in zip(self.nodes,range(len(self.nodes))) if n == None]) != 0 :
            delIndexList = [index for n,index in zip(self.nodes,range(len(self.nodes))) if n == None]
            index = delIndexList[0]
            del(self.nodes[index])
            self.adjMatrix = np.delete(self.adjMatrix, index, axis=0)
            self.adjMatrix = np.delete(self.adjMatrix, index, axis=1)
    def __compareOperator__(self, op1, op2):
        op1Index = self.findEle(self.priorList, op1)
        op2Index = self.findEle(self.priorList, op2)
        return op1Index - op2Index
    
    def findEle(self, eleList, target):
        return eleList[target]
    
    def __dealInvokeChain__(self, chainStr):
        #print(chainStr)
        currentIndex = 0
        stack = []
        pop = 0
        result = []
        while True:    
            eleType, eleContent, nextIndex = self.__nextChainElement__(chainStr, currentIndex)
            #print(eleContent)
            if eleType == None:
                break
            currentIndex = nextIndex
            ele = {}
            ele["type"] = eleType
            ele["content"] = eleContent
            if eleType == "singleEntity":
                result.append(ele)
                return result
            if eleType == "operator":
                if eleContent == ")":
                    while(True):
                        p = stack.pop()
                        pop -= 1
                        if p["content"] != "(":
                            result.append(p)
                        else:
                            break
                    continue
                while pop > 0 and self.__compareOperator__(stack[pop-1]['content'], eleContent) >= 0:
                    if stack[pop-1]["content"] == "(":
                        break
                    result.append(stack.pop())
                    pop -= 1
                stack.append(ele)
                pop += 1
            else:
                result.append(ele)
        stack.reverse()
        result.extend(stack)
        
        return result
    
    def __nextChainElement__(self, line, startIndex):
        if startIndex > len(line)-1:
            return None, None, None
        if line.find("#") < 0:
            return "singleEntity", line, None
        tempIndex = startIndex
        if line[tempIndex] == self.dilema:
            tempIndex += 1
            while line[tempIndex] != self.dilema:
                tempIndex += 1
                if tempIndex > len(line)-1:
                    break
            return "operator",line[startIndex+1:tempIndex], tempIndex+1
        if line[tempIndex] != self.dilema:
            tempIndex += 1
            if tempIndex > len(line)-1:
                return "entity",line[startIndex:tempIndex], tempIndex
            while line[tempIndex] != self.dilema:
                tempIndex += 1
                if tempIndex > len(line)-1:
                    break
            return "entity",line[startIndex:tempIndex], tempIndex
    
    def getAdjMat(self):
        return self.adjMatrix
            
    def getEdges(self):
        return self.edges
    
    def getNodes(self):
        return self.nodes
    
    def getFilePath(self):
        return self.gfp
TopK = None
def setTopK(top):
    global TopK
    TopK = top

if __name__=="__main__":
#    #test GraphStructure
    priorList = {'assign':-2,"return":-2,"param":-2,"~":6 ,'p++':6,"p--":6,"+":4, "-":4, "++":6,"--":6,"*":5 ,"/":5, "%":5, "!=":1, "==":1,
             "<":2,">":2,"<=":2,">=":2,"&&":-2,"&":0,"||":-3,"|": -1,"c_*":6, "c_+":6,"c_-":6,
             "invoke":-3, "parammix":-2, "sizeof":6 ,"<<":3, ">>":3,"!":6, "c_&":6,"c_|":6, "(":7,")":7,"structureaccess":7}
#    graph = GraphStructure(r"E:\tmp\sendGift.txt", 500, 500,"#", priorList)
#    edges = graph.getEdges()
#    nodes = graph.getNodes()
    
    #test Graph
    setTopK(20)
    graph= Graph(r"../1129/980615108947607552_求前缀表达式的值/16电子信息6班-成恩_975587368623472640_980615108947607552.c/",priorList, '#')
    meta = graph.getMetaData()
    subNodes = graph.getSubGraphs()
    #nodes = subNodes[0].getNodes()
    
    adjMat, nodes, edges = graph.getGraphInfo()
    meta = graph.getMetaData()

    print("finish")
    
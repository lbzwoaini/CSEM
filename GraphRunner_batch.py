# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 21:46:55 2019

@author: bzli
"""
import tensorflow as tf
import numpy as np
#from bert_serving.client import BertClient



def initEmbeddingMap(topK):
    embeddingMap = {}
    for i in range(topK):
        embeddingMap["top_"+str(i)] = tf.Variable(tf.random.normal([1, embeddingMapLength],0, 0.01 + 0.002 * i))
    embeddingMap["top_other"] = tf.Variable(tf.random.normal([1, embeddingMapLength],0, 0.01 + 0.002 * topK))
    embeddingMap["onePlaceholder"] = tf.Variable(tf.random.normal([1, embeddingMapLength],0, 0.01 + 0.002 * (topK+1)))
    embeddingMap["startNode"] = tf.Variable(tf.random.normal([1, embeddingMapLength],0, 0.01 + 0.002 * (topK+4)))
    embeddingMap["fuck"] = tf.Variable(tf.random.normal([1, embeddingMapLength],0, 0.01 + 0.002 * (topK+5)))
    #embeddingMap["type_if"] = tf.Variable([1.0]*2)
    #embeddingMap["type_expression"] = tf.Variable([0.0]*2)
    return embeddingMap
    
def lookUpEntityVector(entity):
    global entityEmbeddingMap
    if entity["type"] == "internalResult":
        return entity["content"]
    if entity['content'].startswith("top"):
        if internalEntityVectorHome.get(entity['content']) is not None:
            print("进来了")
            return internalEntityVectorHome.get(entity['content'])
    #entityEmbeddingMap.setdefault(entity['content'], tf.Variable(tf.random.normal([1, embeddingMapLength],0, 0.05)))
    return entityEmbeddingMap[entity['content']]

      
def getChainListsByGraph(graph):
    _ ,nodes, _ = graph.getGraphInfo()
    chainLists = map(lambda x:x["chainList"], nodes)
    return chainLists

def generateTrainStructure(batchGraphs):
    chains = []
    for graph in batchGraphs:
        chains.extend(getChainListsByGraph(graph))
    return chains

def preProcessStack(chains, chainDealedIndexes, stack, shadeList):
    operationList = []
    for chain, dealedIndex, chainIndex, isShade in zip(chains, chainDealedIndexes, range(len(chainDealedIndexes)), shadeList):
        if isShade == 1:
            operationList.append({"content":"noOp", "type":"operator"})
            continue
        if dealedIndex >= len(chain):
            operationList.append({"content":"noOp", "type":"operator"})
            shadeList[chainIndex] = 1
            continue
        for chainElement in chain[dealedIndex:]:
            myType = chainElement["type"]
            if  myType == "entity":
                stack[chainIndex].append(chainElement)
                chainDealedIndexes[chainIndex] += 1
                continue
            elif myType == "singleEntity":
                stack[chainIndex].append(chainElement)
                chainDealedIndexes[chainIndex] += 1
                operationList.append({"content":"single", "type":"operator"})
                break
            else:
                operationList.append(chainElement)
                chainDealedIndexes[chainIndex] += 1
                break
    assert len(operationList) == len(chains)
    return operationList

def calcateChainBatch(chains, gru, keep_prob):
    calcateStack=[[] for i in range(len(chains))]
    indexes= [0] * len(chains)
    shadeList = [0] * len(chains)
    while True:
        #len(operationList) == len(calcateStack)
        operationList = preProcessStack(chains, indexes, calcateStack, shadeList)
        if len([i for i in shadeList if i == 0]) == 0:
            break
        else:
            calcateOneStepBatch(calcateStack, operationList, gru, keep_prob, shadeList)
    return calcateStack

#处理singleEntity计算遮蔽
def dealSingleEntityVector(calcateStack, operations, shadeList):
    singleEntityOpIndexes = [index for op, index in zip(operations, range(len(operations))) if op['content'] == 'single']
    for index in singleEntityOpIndexes:
        if shadeList[index] == 1:
            continue
        stack = calcateStack[index]
        entity = stack.pop()
        stack.append({"type":"internalResult", 'content':lookUpEntityVector(entity)})
        shadeList[index] = 1

def calcateOneStepBatch(calcateStack, operations, gru, keep_prob, shadeList):
    global operatorType
    #遮蔽singelEntity
    dealSingleEntityVector(calcateStack, operations, shadeList)
    validCalcateStacks = [stack for stack , isShade in zip(calcateStack, shadeList) if isShade == 0]
    validOperations = [op for op, isShade in zip(operations, shadeList) if isShade == 0]
    assert len(validCalcateStacks) == len(validOperations)
    op1s = []
    op2s = []
    for stack, op in zip(validCalcateStacks, validOperations):
        #print(stack)
        chainStack = stack

        opType = operatorType[op['content']]

        if opType == 1:
            op1s.append(lookUpEntityVector(chainStack.pop()))
            op2s.append(lookUpEntityVector({"type":"entity", "content":"onePlaceholder"}))
        else:
            op1s.append(lookUpEntityVector(chainStack.pop()))
            op2s.append(lookUpEntityVector(chainStack.pop()))
    results = gru.calcate(op1s, op2s, validOperations, keep_prob)
    for stack, result in zip(validCalcateStacks, results):
        stack.append({"type":"internalResult", 'content':tf.reshape(result, shape = [1, result.shape[-1]])})
   
def dealGraphs(graphs, gru, keep_prob): 
    global entityEmbeddingMap
    #print(graphs[0].getMethodFilePath())
    calcateResultStacks = calcateChainBatch(generateTrainStructure(graphs), gru, keep_prob)
    assert len([i for i in calcateResultStacks if len(calcateResultStacks) != 1])>0
    resultList = []
    for st in calcateResultStacks:
        resultList.append(st.pop()['content'])
    #按照每个graph的chain数量将这一批chain还原
    graphEmbeddings = []
    chainNumList = []
    for graph in graphs:
        _,nodes,_ = graph.getGraphInfo()
        chainNumList.append(len(nodes))
    assert sum(chainNumList) == len(resultList)
    currentResultIndex = 0
    for num in chainNumList:
        graphEmbeddings.append(tf.convert_to_tensor(resultList[currentResultIndex:currentResultIndex + num]))
        currentResultIndex += num
    return graphEmbeddings

def setTopK(top):
    global TopK
    TopK = top
def setEmbeddingMapLength(length):
    global embeddingMapLength, entityEmbeddingMap
    embeddingMapLength = length
    entityEmbeddingMap = initEmbeddingMap(TopK)
operatorType = {'assign':2,"return":1,"param":2,"~":1 ,'p++':1,"p--":1,"+":2, "-":2, "++":1,"--":1,"*":2 ,"/":2, "%":2, "!=":2, "==":2,
             "<":2,">":2,"<=":2,">=":2,"&&":2,"&":2,"||":2,"|": 2,"c_*":1, "c_+":1,"c_-":1,"^":1,
             "invoke":1, "parammix":2, "sizeof":1 ,"<<":2, ">>":2,"!":1, "c_&":1,"c_|":1, "structureaccess":2}
internalEntityVectorHome = {}
TopK = None
embeddingMapLength = None
currentGraph = None
entityEmbeddingMap = None
if __name__=="__main__":
    setTopK(10)
    setEmbeddingMapLength(64)

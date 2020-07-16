# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 11:57:58 2019

@author: bzli
"""
import tensorflow as tf

class EventEmbedTransformer:
    def __init__(self, headNum, 
                 inputLength, 
                 operationList,
                 perceptionLayerNum, 
                 perceptionLayerUnitNumList, 
                 activateFunction):
        self.headNum = headNum
        self.inputLength = inputLength
        self.operationNum = len(operationList)
        self.operationList = operationList
        self.perceptionLayerNum = perceptionLayerNum
        self.perceptionUnitNumList = perceptionLayerUnitNumList
        self.activateFunction = activateFunction
        self.trainableVariables = []
        self.__initTensorSelector__()
        self.__initWeight__()
        
    def __initTensorSelector__(self):
        self.tensorMap = {}
        index = 0
        for operation in self.operationList:
            tempList = []
            tempList.append(index)
            index += 1
            tempList.append(index)
            index += 1
            self.tensorMap[operation] =  tempList
            
    def __initWeight__(self):
        self.operationTensor = tf.Variable(tf.random.normal([2*self.operationNum, self.headNum, self.inputLength, self.inputLength],mean = 0, stddev = 0.01), name = "tensor_operations")
        self.fullyConnectWeight = []
        self.fullyConnectedB = []
        for i in range(self.perceptionLayerNum):
            unitNum = self.perceptionUnitNumList[i]
            if i == 0:
                W = tf.Variable(tf.random.normal([self.headNum * self.inputLength * 2, unitNum], 0, 0.01), name = "perception_W_" + str(i))
                #self.trainableVariables.append(W)
            else:
                W = tf.Variable(tf.random.normal([self.perceptionUnitNumList[i-1], unitNum], 0, 0.01), name = "perception_W_" + str(i))
                #self.trainableVariables.append(W)
            b = tf.Variable(tf.zeros([unitNum]), name = "perception_b_" + str(i))
            self.fullyConnectWeight.append(W)
            self.fullyConnectedB.append(b)
    
    def embed(self, entity1Vecs, entity2Vecs, operations, keep_prob):
        opTensorsIndex = []
        for op in operations:
           opTensorsIndex.append(self.tensorMap[op['content']])
        oprationTensor = tf.nn.embedding_lookup(self.operationTensor, opTensorsIndex)
        newEntity1Vecs = tf.reshape(entity1Vecs, shape = [len(entity1Vecs), 1, 1, entity1Vecs.shape[-1]])
        newEntity2Vecs = tf.reshape(entity2Vecs, shape = [len(entity2Vecs), 1, 1, entity2Vecs.shape[-1]])
        result1 = tf.matmul(newEntity1Vecs, oprationTensor[:,0,:,:,:])
        result2 = tf.matmul(newEntity2Vecs, oprationTensor[:,1,:,:,:])
        
        #实时调试看
        finalVec = tf.concat([result1, result2], axis = 1)
        x = tf.reshape(finalVec, [len(entity1Vecs), -1])
        #print("finalVec", x.shape)
        for i in range(self.perceptionLayerNum):
            z = tf.matmul(x, self.fullyConnectWeight[i])+self.fullyConnectedB[i]
            z = self.activateFunction(z)
            z = tf.nn.dropout(z,keep_prob=keep_prob)
            x = z
        return x
    
    def getTrainableVariables(self):
        pass

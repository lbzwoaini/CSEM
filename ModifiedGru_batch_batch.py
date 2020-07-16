# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 19:20:08 2019

@author: bzli
"""
import tensorflow as tf
from EventEmbeddingTransforrmer_batch import EventEmbedTransformer
class ModifiedGru:
    def __init__(self, gruUnitNum, transformerHeadNum, transformerInputLength, 
                 transfoemerPerceptionLayerNum, transformerPerceptionLayerUnitNumList, transformerActivateFunction, operationList):

        self.gruUnitNum = gruUnitNum
        self.transformerInputLength = transformerInputLength
        self.transformerHeadNum = transformerHeadNum
        self.transformerPerceptionLayerNum = transfoemerPerceptionLayerNum
        self.transformerLayerUnitNumList = transformerPerceptionLayerUnitNumList
        self.transformerActivateFunction = transformerActivateFunction
        self.operationList = operationList
        self.transformerOPerationNum = len(self.operationList)
        self.__initWeight__()
        
    def __initWeight__(self):
        #重置门参数
        self.Wr = tf.Variable(tf.random.normal([2 * self.transformerInputLength, self.gruUnitNum],0, 0.01), name = "reset_gate_W" )
        self.br = tf.Variable(tf.zeros([1, self.gruUnitNum], dtype = tf.float32), name = "reset_gate_b")
        
        
        #更新门参数
        self.Wz = tf.Variable(tf.random.normal([2 * self.transformerInputLength, self.gruUnitNum],0, 0.01), name = "update_gate1_W" )
        self.bz = tf.Variable(tf.zeros([1, self.gruUnitNum], dtype = tf.float32), name = "update_gate1_b")
        #输出门部分
        self.transformer = EventEmbedTransformer(headNum = self.transformerHeadNum, 
                                                  operationList = self.operationList,
                                                  inputLength = self.transformerInputLength, 
                                                  perceptionLayerNum = self.transformerPerceptionLayerNum, 
                                                  perceptionLayerUnitNumList = self.transformerLayerUnitNumList, 
                                                  activateFunction = self.transformerActivateFunction)
        self.Wo = tf.Variable(tf.random.normal([2 * self.transformerInputLength, self.gruUnitNum],0, 0.01), name = "reset_gate2_W" )
        self.bo = tf.Variable(tf.zeros([1, self.gruUnitNum], dtype = tf.float32), name = "reset_gate2_b")
        
    def calcate(self, op1s, op2s, operations, keep_prob):
        t_op1s = tf.convert_to_tensor(op1s)
        t_op2s = tf.convert_to_tensor(op2s)
#        print(t_op1s.shape)
#        print(t_op2s.shape)
        newOp1s = tf.reshape(t_op1s, shape = [len(op1s), -1])
        newOp2s = tf.reshape(t_op2s, shape = [len(op2s), -1])
        concatedInput = tf.concat([newOp1s, newOp2s], axis = 1)
        self.rtGateOut = tf.nn.sigmoid(tf.matmul(concatedInput, self.Wr) +self.br)
        self.updateGateOut = tf.nn.sigmoid(tf.matmul(concatedInput, self.Wz) +self.bz)
        self.dealdUpdateGateOut = 1 - self.updateGateOut

        out = newOp1s * self.dealdUpdateGateOut + self.updateGateOut * (tf.matmul(tf.concat([self.rtGateOut * newOp1s, newOp2s], axis = 1), self.Wo)+self.bo)
        
        #self.transformer.embed(self.rtGateOut * newOp1s, newOp2s, operations, keep_prob)
        return out
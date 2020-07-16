# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 10:01:47 2019

@author: bzli
"""
import numpy as np
import tensorflow as tf
import GraphRunner_batch
from ModifiedGru_batch_batch import ModifiedGru 
#from GAT import GAT
import process
from layers import AttenHead
from base_gattn import BaseGAttN
import pickle
class GAT(BaseGAttN):
    def __init__(self, nb_classes, hid_units, n_heads, maxNodes, activation=tf.nn.elu, residual=False):
        self.nb_classes = nb_classes
        self.hid_units = hid_units
        self.n_heads = n_heads
        self.activation = activation
        self.residual = residual
        self.firstHeads = []
        for _ in range(self.n_heads[0]):
            self.firstHeads.append(AttenHead(maxNodes))
        self.middleHeads = []
        self.middleHeads.append("placeholder")
        for i in range(1, len(hid_units)):
            self.middleHeads.append([])
            for _ in range(n_heads[i]):
                self.middleHeads[i].append(AttenHead(maxNodes))
        self.lastHeads = []
        for i in range(n_heads[-1]):
            self.lastHeads.append(AttenHead(maxNodes))
            
    def inference(self, inputs,  bias_mat, attn_drop, ffd_drop):
        attns = []
        for i in range(self.n_heads[0]):
            attns.append(self.firstHeads[i].calcate(inputs, bias_mat=bias_mat,
                out_sz=self.hid_units[0], activation=self.activation,
                in_drop=ffd_drop, coef_drop=attn_drop, residual=self.residual))
        #一个节点的K个注意力输出的连接
        h_1 = tf.concat(attns, axis=-1)
        
        #中间层定义
        for i in range(1, len(self.hid_units)):
            h_old = h_1
            attns = []
            for j in range(self.n_heads[i]):
                attns.append(self.middleHeads[i][j].calcate(h_1, bias_mat=bias_mat,
                out_sz=self.hid_units[i], activation=self.activation, in_drop=ffd_drop, coef_drop=attn_drop, residual=self.residual))
            h_1 = tf.concat(attns, axis=-1)
        out = []
        
        #最后一层的结构，均值输出
        for i in range(self.n_heads[-1]):
            out.append(self.lastHeads[i].calcate(h_1, bias_mat=bias_mat,
                out_sz=self.nb_classes, activation=lambda x: x,
                in_drop=ffd_drop, coef_drop=attn_drop, residual=self.residual))
        logits = tf.add_n(out, name="head_add") / self.n_heads[-1]
    
        return logits

class Model:

    def __init__(self, priorList, entityEmbeddingMap,
                 gruInputLength, gruUnitNum, transformerHeadNum, transformerPerceptionLayerUnitNumList, 
                 transformerActivateFunction, operationList, gatHiddenUnits, nbClasses, headNum, classNum, maxNodes, hammingLength):
        self.priorList = priorList
        self.entityEmbeddingMap = entityEmbeddingMap
        #self.tensorSelectorMap = tensorSelectorMap
        #self.entityMap = entityMap
        self.gruInputLength = gruInputLength
        self.gru = ModifiedGru(gruUnitNum = gruUnitNum, transformerHeadNum = transformerHeadNum, transformerInputLength = gruInputLength, 
                 transfoemerPerceptionLayerNum = len(transformerPerceptionLayerUnitNumList), transformerPerceptionLayerUnitNumList = transformerPerceptionLayerUnitNumList, 
                 transformerActivateFunction = transformerActivateFunction, operationList = operationList)
        
        self.gatHiddenUnits = gatHiddenUnits
        self.nbClasses = nbClasses
        self.headNum = headNum
        self.maxNodes = maxNodes
        self.gat = GAT(nb_classes = self.nbClasses,
                       hid_units = self.gatHiddenUnits, 
                       n_heads = [8,1], 
                       activation=tf.nn.elu,
                       maxNodes = maxNodes,
                       residual=False)
        self.outConv = tf.layers.Conv1D(32,1, use_bias = False)
        self.outDense = tf.layers.Dense(1400, activation=tf.nn.elu)
        self.outDense2 = tf.layers.Dense(700, activation=tf.nn.elu)
        self.outDense3 = tf.layers.Dense(128, activation=tf.nn.elu)
        self.outDense4 = tf.layers.Dense(classNum, activation=tf.nn.elu)
        self.hammingWeight = tf.Variable(tf.random.normal([128, hammingLength],0,0.01))
        self.hammingBasis = tf.Variable(tf.zeros([hammingLength]))
        self.etOutMap = {}
        self.gatOutMap = {}
        #self.outDense2 = tf.layers.Dense(classNum, activation=tf.nn.elu)
        
    def __setMap__(self, m, keys, values):
        for key, value in zip(keys, values):
            m[key.getMethodFilePath()] = [key,value]
            
    def saveMaps(self, map1Path, map2Path):
        pickle.dump(self.etOutMap, open(map1Path, 'wb'))
        pickle.dump(self.gatOutMap, open(map2Path, 'wb'))
        
    def paddingEmbeddingGraphs(self, embeddingGraphs, adjMats):
        maxNodeNum = max([len(ge) for ge in embeddingGraphs])
        resultEmbedidngGraphs = []
        resultAdjMats = []
        for graphEmbedding, adjMat in zip(embeddingGraphs, adjMats):
            g = tf.reshape(graphEmbedding, shape = [len(graphEmbedding), -1])
            g = tf.pad(g, [[0,maxNodeNum - len(graphEmbedding)],[0, 0]])
            resultEmbedidngGraphs.append(g)
            m = tf.pad(adjMat, [[0, maxNodeNum - len(adjMat)],[0, maxNodeNum - len(adjMat)]])
            resultAdjMats.append(m)
        return tf.convert_to_tensor(resultEmbedidngGraphs), tf.convert_to_tensor(resultAdjMats)
        
    def inference(self, batchGraphs, attnDrop, ffdDrop, keep_prob):


        
        #print("start event embedding------------------------------")
        graphResults = GraphRunner_batch.dealGraphs(batchGraphs, self.gru, keep_prob)
        if keep_prob == 1:
            self.__setMap__(self.etOutMap, batchGraphs, graphResults)
        #print("finish event embedding-----------------------------")
        adjMats = []
        #收集邻接矩阵
        for graphEmbedding, graph in zip(graphResults, batchGraphs):
            adjMat, nodes, edges = graph.getGraphInfo()
            adjMats.append(adjMat)
        padGraphEmbeddings, padAdjMats = self.paddingEmbeddingGraphs(graphResults, adjMats)
        
        biasMats = process.adj_to_bias(padAdjMats, [len(padAdjMats[0])] * len(padAdjMats))
        a = padGraphEmbeddings#tf.reshape(graphEmbedding,shape=[1,-1,graphEmbedding.shape[-1]])
        #print("start GAT:-----------------------")
        logits = self.gat.inference(inputs = a, bias_mat = biasMats, attn_drop=attnDrop, ffd_drop=ffdDrop)
        if keep_prob == 1:
            self.__setMap__(self.gatOutMap, batchGraphs, logits)        
        
        #print("finish GAT--------------------------------")
        logits1 = self.outConv(logits)
        logits2 = tf.reshape(logits1, shape=[len(logits1), -1])
##        try:
        logits3 = tf.pad(logits2, [[0,0],[0,32*self.maxNodes - logits2.shape[1]]])
##        except Exception  as e:
##            print(e)
#        resultLogits = []
#        for index in range(int(len(logits3)/2)):
#            resultLogits.append(tf.concat([logits3[index], logits3[index + (int(len(logits3)/2)-1)]], axis=0))
#        logits4 = self.outDense4(self.outDense3(self.outDense2(self.outDense(tf.convert_to_tensor(resultLogits)))))
        return self.outDense3(logits3)
    
    def get_cos_distance(self, q, a):
        # calculate cos distance between two sets
        # more similar more big
        pooled_len_1 = tf.sqrt(tf.reduce_sum(q * q, 1))
        pooled_len_2 = tf.sqrt(tf.reduce_sum(a * a, 1))
        pooled_mul_12 = tf.reduce_sum(q * a, 1)
        score = tf.div(pooled_mul_12, pooled_len_1 * pooled_len_2 +1e-8, name="scores")
        return score
    
    def get_ou_distance(self, logits1, logits2):
        return tf.sqrt(tf.reduce_sum(tf.square(logits1 - logits2), axis = 1))
#    def oushiLoss(self, q, a):
#        tf.reduce_sum(tf.square(tf.abs(q-a)), axis=1)
#        return score
    def hammingLoss(self, logits1, logits2, labels):
#        hamming1 = tf.sign(tf.matmul(logits1, self.hammingWeight) + self.hammingBasis)
#        hamming2 = tf.sign(tf.matmul(logits2, self.hammingWeight) + self.hammingBasis)
#        floatLabel = tf.convert_to_tensor(labels, dtype = tf.float32)
#        oneLike = tf.ones_like(floatLabel)
#        signLabel = tf.where(tf.equal(floatLabel, oneLike), floatLabel, tf.ones([len(labels)])*-1)
#        temp = signLabel - (tf.reduce_sum((tf.sign(hamming1) * tf.sign(hamming2))/len(self.hammingWeight[0])))
#        temp = tf.square(temp)
#        tempLeft = tf.abs(signLabel) * temp
#        loss = tf.reduce_mean(tempLeft)
        #temp = self.get_cos_distance(logits1, logits2)
        temp = self.get_cos_distance(logits1, logits2)
        ls = tf.convert_to_tensor(labels)
        posiDistance = tf.abs(temp[ls.numpy() == 1])
        negaDistance = tf.abs(temp[ls.numpy() == 0])
        
        loss = tf.reduce_sum(tf.maximum(0, 1 - posiDistance + negaDistance))
#        positiveLoss = 0
#        negativeLiss = 0
#        p = 0
#        n = 0
#        for label,index in zip(labels,range(len(labels))):
#            if label == 1:
#                positiveLoss += 1-temp[index]
#                p+=1
#            else:
#                negativeLiss += temp[index]
#                n+=1
        return loss
    
    def hammingRepresentation(self, logits1, logits2):
        hamming1 = tf.sign(tf.matmul(logits1, self.hammingWeight) + self.hammingBasis)
        hamming2 = tf.sign(tf.matmul(logits2, self.hammingWeight) + self.hammingBasis)
        return hamming1, hamming2
    
    def hammingDistance(self, samples1, samples2):
        distances = tf.reduce_mean((samples1 - samples2)/4, axis = 1)
        return distances
    
    def cloneLoss(logits, labels, classNum, gamma):
        softmax = tf.reshape(tf.nn.softmax(logits), [-1])  # [batch_size * n_class]
        labels = tf.range(0, logits.shape[0]) * logits.shape[1] + labels
        prob = tf.gather(softmax, labels)
        weight = tf.pow(tf.subtract(1., prob), gamma)
        loss = -tf.reduce_mean(tf.multiply(weight, tf.log(prob)))
        equal = tf.equal(tf.argmax(tf.nn.softmax(logits), 1), tf.argmax(tf.one_hot(labels, classNum), 1))
        faultClassifyIndexes = [index for ele,index in zip(equal, range(len(equal))) if ele.numpy() == False]
        return loss,None, None, faultClassifyIndexes 
    
    def normalLoss(logits, labels, classNum):
        prediction = tf.argmax(tf.nn.softmax(logits), 1)
        TP = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(labels, tf.ones_like(prediction)),tf.equal(prediction, tf.ones_like(labels))), dtype=tf.float32))
        FN = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(labels, tf.ones_like(labels)),tf.equal(prediction, tf.zeros_like(prediction))),dtype=tf.float32))
        FP = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(labels, tf.zeros_like(labels)),tf.equal(prediction, tf.ones_like(prediction))),dtype=tf.float32))
        R = TP/(TP+FN)
        P = TP/(TP+FP)
        F1 = (2 * P * R)/(P + R)
#        if len(logits) != 1:
#            print("mean_loss:",tf.reduce_mean(loss))
#        if tf.reduce_mean(loss)>50:
            
#            print("logits", logits)
#            print("onehotL:",oneHotLabels)
#            print("lossdetail",loss)
#            print("mean_loss:",tf.reduce_mean(loss))
#             print("max loss: ",loss.numpy().max())
        #tf.argmax(loss, dimension = 0), loss[tf.argmax(loss, dimension = 0).numpy()[0]]
        return F1, P, R, prediction
    def compute_focal_loss(logits,labels,alpha=tf.constant([[0.5],[0.5]]),class_num=2,gamma=2):
        '''
        :param logits:
        :param labels:
        :return:
        '''
        labels = tf.reshape(labels, [-1])
        labels = tf.cast(labels,tf.int32)
        labels = tf.one_hot(labels, class_num, on_value=1.0, off_value=0.0)
        pred = tf.nn.softmax(logits)
     
        temp_loss = -1*tf.pow(1 - pred, gamma) * tf.log(pred)
     
        focal_loss =  tf.reduce_mean(tf.matmul(temp_loss * labels,alpha))
     
        return focal_loss

    def focal_loss(logits,labels,class_num, gamma, alpha=tf.constant([[0.8],[0.2]]),):
        
        '''
        :param logits:  [batch_size, n_class]
        :param labels: [batch_size]
        :return: -(1-y)^r * log(y)
        '''
        labels = tf.reshape(labels, [-1])
        labels = tf.cast(labels,tf.int32)
        labels = tf.one_hot(labels, class_num, on_value=1.0, off_value=0.0)
        pred = tf.nn.softmax(logits)
     
        temp_loss = -1*tf.pow(1 - pred, gamma) * tf.log(pred)
     
        focal_loss =  tf.reduce_mean(tf.matmul(temp_loss * labels,alpha))
        return focal_loss,None, None, None

    def loss(logits, labels, nb_classes, class_weights):
        sample_wts = tf.reduce_sum(tf.multiply(tf.one_hot(labels, nb_classes), class_weights), axis=-1)
        xentropy = tf.multiply(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits), sample_wts)
        return tf.reduce_sum(xentropy, name='xentropy_mean')

    def l2(loss, lr, l2_coef):
        # weight decay
        vars = tf.trainable_variables()
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if v.name not in ['bias', 'gamma', 'b', 'g', 'beta']]) * l2_coef
        return lossL2
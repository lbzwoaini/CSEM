# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 22:19:41 2019

@author: bzli
 """
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
from Model_batch import Model
import random
import os
from GraphStructure import Graph
import GraphStructure
import pickle
import GraphRunner_batch
import random
import datetime
#def splitDataSet():
#    global trainPercen, classNum
#    trainSetSample = []
#    trainSetLabel = []
#    testSetSample = []
#    testSetLabel = []
#    labelArray = np.array(labelList)
#    for classLabel in range(max(labelList)+1):
#        res = np.where(labelArray == classLabel)
#        trainNum = int(len(res[0]) * trainingPercen)
#        for i in range(trainNum):
#            trainSetSample.append(graphList[int(res[0][i])])
#            trainSetLabel.append(labelList[int(res[0][i])])
#        for i in range(trainNum, len(res[0])):
#            #print(i)
#            testSetSample.append(graphList[int(res[0][i])])
#            testSetLabel.append(labelList[int(res[0][i])])
#    return trainSetSample, trainSetLabel, testSetSample, testSetLabel
            

#璇诲彇鏁版嵁闆嗗苟澶勭悊涓篏raph鏂囦欢, 鏍囩鏂囦欢

#def saveFaultClassifySample(samples, faultList):
#    for index in faultList:
#        sampleName = samples[index].getMethodFilePath()
#        if faultClassifySampleMap.get(sampleName) == None:
#            faultClassifySampleMap[sampleName] = 1
#        else:
#            faultClassifySampleMap[sampleName] += 1
            
#def generatePart(samples, labels):
#    pairList = []
#    labelList = []
#    groupMap = {}
#    #对样本进行分组
#    for sample,label in zip(samples, labels):
#        if groupMap.get(label) is None:
#            groupMap[label] = []
#        else:
#            groupMap[label].append(sample)
#    print("样本分组完成")
#    #生成正例
#    for label, sam in groupMap.items():
#        for i in range(1, len(sam)):
#            pairList.append([samples[i-1],samples[i]])
#            labelList.append(1)
#            if len(labelList) >= 50:
#                break
#        if len(labelList) >= 50:
#            break
#    print("反例生成完成",len(labelList), "个")
#    #生成7%的反例
#    sn = int(len(labelList) / 0.07)#len(labelList) * 2#
#    antiSamples = []
#    antiLabels = []
#    print("开始生成正例", sn, "个")
#    while True:
#        try:
#            in1 = random.randint(0, max(labels))
#            in2 = random.randint(0, max(labels))
#            if in1==in2:
#                continue
#            c1Samples = groupMap[in1]
#            c2Samples = groupMap[in2]
#            sin1 = random.randint(0, len(c1Samples)-1)
#            sin2 = random.randint(0, len(c2Samples)-1)
#            antiSamples.append([c1Samples[sin1],c2Samples[sin2]])
#            antiLabels.append(0)
#            #print("antiSamples:",len(antiLabels))
#            if len(antiSamples) >= sn:
#                break
#        except Exception as e:
#            continue
#    pairList.extend(antiSamples)
#    labelList.extend(antiLabels)
#    return pairList, labelList

#def generatrCloneDataSet(trainSamples, trainLabels, testSamples, testLabels):
#    trainSamples, trainLabels = generatePart(trainSamples, trainLabels)
#    testSamples, testLabels = generatePart(testSamples, testLabels)
#    return trainSamples, trainLabels, testSamples, testLabels


#def shuffleAllDataSet():
#    global graphList, labelList
#    shuffle = list(range(len(graphList)))
#    random.shuffle(shuffle)
#    shuffledSamples = []
#    shuffledLabels = []
#    for index in shuffle:
#        shuffledSamples.append(graphList[index])
#        shuffledLabels.append(labelList[index])
#    graphList = shuffledSamples
#    labelList = shuffledLabels
def getNextTrainBatch():
    global nowTrainIndex
    global batchSize
    if nowTrainIndex >= len(trainSamples):
        return None, None
    if nowTrainIndex + batchSize > len(trainSamples):
        sample = trainSamples[nowTrainIndex:]
        label = trainLabels[nowTrainIndex:]
        nowTrainIndex += batchSize
        return sample, label
    sample = trainSamples[nowTrainIndex:nowTrainIndex+batchSize]
    label = trainLabels[nowTrainIndex:nowTrainIndex+batchSize]
    nowTrainIndex += batchSize
    return sample, label

def getNextTestBatch():
    global nowTestIndex
    global testBatchSize
    if nowTestIndex >= len(testSamples):
        return None, None
    if nowTestIndex + testBatchSize > len(testSamples):
        sample = testSamples[nowTestIndex:]
        label = testLabels[nowTestIndex:]
        nowTestIndex += testBatchSize
        return sample, label
    sample = testSamples[nowTestIndex:nowTestIndex+testBatchSize]
    label = testLabels[nowTestIndex:nowTestIndex+testBatchSize]
    nowTestIndex += testBatchSize
    return sample, label

def resetIndex():
    global nowTrainIndex, nowTestIndex
    nowTrainIndex = 0
    nowTestIndex = 0

def shuffleSet(trainSamples, trainLabels):
    shuffle = list(range(len(trainSamples)))
    random.shuffle(shuffle)
    shuffledSamples = []
    shuffledLabels = []
    for index in shuffle:
        shuffledSamples.append(trainSamples[index])
        shuffledLabels.append(trainLabels[index])
    return shuffledSamples, shuffledLabels
    
def loadData(p):
    graphList = pickle.load(open(p + "/train_samples.pickle", 'rb'))
    labelList = pickle.load(open(p + "/train_labels.pickle",'rb'))
    graphList2 = pickle.load(open(p + "/test_samples.pickle",'rb'))
    labelList2 = pickle.load(open(p + "/test_labels.pickle",'rb'))
    #negativeSamplesMap = pickle.load(open("data/negative_samples.pickle", 'rb'))
    negMap = pickle.load(open(p + "/negaMap.pickle",'rb'))
    print("load samples:", str(len(graphList)))
    return graphList, labelList, graphList2, labelList2, negMap
    

def getcolumn(data, index):
    res = []
    for i in data:
        res.append(i[index])
    return res

def generateNegativeSamples(nm, originalSamples):
#    if epo %5 ==0:
#        negMap.clear()
    negaList = []
    for sample in originalSamples:
        if negMap.get(sample[0].getMethodFilePath()) is not None:
            negaList.append(negMap.get(sample[0].getMethodFilePath()))
            continue
        print(sample[0].getMethodFilePath())
        print("nega error")
        #clsName = sample[0].getClassName()
        #validNegativeSamples = nm[clsName]
        #validNegativeSamples = [vali for vali in validNegativeSamples if vali[0].getMethodFilePath() == sample[0].getMethodFilePath() or vali[1].getMethodFilePath() == sample[0].getMethodFilePath()]
        #ind = random.randint(0, len(validNegativeSamples)-1)
        #negMap[sample[0].getMethodFilePath()] = validNegativeSamples[ind]
        #negaList.append(validNegativeSamples[ind])
    return negaList
    

negMap = {}
negativeSamplePolicy = "random"    
batchSize = 64
testBatchSize = 64
#trainingPercen = 0.8
epoch = 200
lr = 0.001
#foldNum = 15
classNum = 2
TopK = 10
embeddingMapLength = 128
readPickle = False

priorList = {'assign':-5,"return":-5,"param":-2,"~":6 ,'p++':6,"p--":6,"+":4, "-":4, "++":6,"--":6,"*":5 ,"/":5, "%":5, "!=":1, "==":1,
             "<":2,">":2,"<=":2,">=":2,"&&":-2,"&":0,"||":-3,"|": -1,"c_*":6, "c_+":6,"c_-":6,"^":6,
             "invoke":-4, "parammix":-3, "sizeof":6 ,"<<":3, ">>":3,"!":6, "c_&":6,"c_|":6, "(":7,")":7,"structureaccess":7}
GraphStructure.setTopK(TopK)
GraphRunner_batch.setTopK(TopK)
GraphRunner_batch.setEmbeddingMapLength(embeddingMapLength)
#dataFilePath = "E:/program_embedding_literature/dealed_function/"
dataFilePath = "../1129/"
if readPickle:
    print("load data from pickle")
    trainSamples = pickle.load(open("trainSamples.pickle"+str(classNum),'rb'))
    trainLabels = pickle.load(open("trainLabels.pickle"+str(classNum),'rb'))
    testSamples = pickle.load(open("testSamples.pickle"+str(classNum),'rb'))
    testLabels = pickle.load(open("testLabels"+str(classNum),'rb'))
else:

    originalTrainSamples, originalTrainLabels, testSamples, testLabels, negMap = loadData("data") 
    trainSamples = [s for s,label in zip(originalTrainSamples,originalTrainLabels) if label == 1]
    trainLabels = [label for label in originalTrainLabels if label == 1]
    
    #trainSamples, trainLabels, testSamples, testLabels = generatrCloneDataSet(trainSamples, trainLabels, testSamples, testLabels)
    pickle.dump(trainSamples, open("trainSamples.pickle"+str(classNum),'wb'))
    pickle.dump(trainLabels, open("trainLabels.pickle"+str(classNum),'wb'))
    pickle.dump(testSamples, open("testSamples.pickle"+str(classNum),'wb'))
    pickle.dump(testLabels, open("testLabels"+str(classNum),'wb'))

print("positive in training set:", len([label for label in trainLabels if label == 1]))
print("negative in training set:", len([label for label in trainLabels if label == 0]))
print("positive in test set:", len([label for label in testLabels if label == 1]))
print("negative in test set:", len([label for label in testLabels if label == 0]))
nowTrainIndex = 0
nowTestIndex = 0
model = Model(priorList = priorList, entityEmbeddingMap = None, gruUnitNum = 128, transformerHeadNum = 2 , gruInputLength = 128,
                 transformerPerceptionLayerUnitNumList = [128], transformerActivateFunction = tf.nn.elu, 
                 operationList = list(priorList.keys()), gatHiddenUnits = [64], nbClasses =128, headNum = 2, classNum = classNum, maxNodes = 170, hammingLength = 16)
optimizer = tf.train.AdamOptimizer(lr)

historyMeanAccurancy = []
increSameNums = 4
increBatchSize = False
faultClassifySampleMap = {}
th = 5
for epo in range(1, epoch + 1):
    print("epoch:", epo)
    print("=============================================")
    meanTrainBeforeLoss = []
    meanTrainAfterLoss = []
    meanTrainAccurancy = []
    meanAccurancy = []
    resetIndex()
    
    a = datetime.datetime.now()
    
    trainSamples, trainLabels = shuffleSet(trainSamples, trainLabels)
    #if epo % 8 == 0:
    #    batchSize += 8
    step = 0
    maps = []
    while True:
        step += 1
        samples, labels = getNextTrainBatch()
        if samples is None:
            break
        negativeSamples = generateNegativeSamples(None, samples)
        with tf.GradientTape() as tape:
            samples.extend(negativeSamples)
            labels.extend([0]*len(negativeSamples))
            samples, labels = shuffleSet(samples, labels)
            samples1 = getcolumn(samples,0)
            samples2 = getcolumn(samples, 1)
            
            logits1 = model.inference(samples1, attnDrop = 0, ffdDrop = 0, keep_prob=1)
            logits2 = model.inference(samples2, attnDrop = 0, ffdDrop = 0, keep_prob=1)
            
            loss = model.hammingLoss(logits1, logits2, labels)#,classNum, 2)
        meanTrainBeforeLoss.append(loss)
        variables = tape.watched_variables()

        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
#        logits1 = model.inference(samples1, attnDrop = 0, ffdDrop = 0, keep_prob=1)
#        logits2 = model.inference(samples2, attnDrop = 0, ffdDrop = 0, keep_prob=1)
#        loss = model.hammingLoss(logits1, logits2, labels)#,classNum, 2)
#        meanTrainAfterLoss.append(loss)
        
    b = datetime.datetime.now()
    print("mean training before loss: ", tf.reduce_mean(meanTrainBeforeLoss))
#    print("mean training after loss: ", tf.reduce_mean(meanTrainAfterLoss))
    #娴嬭瘯闃舵杈撳嚭
    lll=0
    testSampleNums = 0
    accurancy = 0
    TP = 0
    FN = 0
    FP = 0
    maps = []
    if epo % 1== 0 or epo == 1:
        while True:
            samples, labels = getNextTestBatch()
            if samples is None:
                break
            resultMap = {}
            samples1 = getcolumn(samples,0)
            samples2 = getcolumn(samples, 1)
            logits1 = model.inference(samples1, attnDrop = 0, ffdDrop = 0, keep_prob=1)
            logits2 = model.inference(samples2, attnDrop = 0, ffdDrop = 0, keep_prob=1)
            
            for l1, l2, s1, s2, label in zip(logits1,logits2, samples1, samples2, labels):
                maps.append({"l1":l1,"l2":l2, "s1":s1, "s2":s2,"label":label})
        pickle.dump(maps, open("result.pickle"+str(epo), 'wb'))
        model.saveMaps("EtMap.pcikle"+str(epo), "GatMap.pickle"+str(epo))
    c = datetime.datetime.now()
    k = b - a
    k2 = c - b
    k3 = c - a
    print("train seconds:", k.total_seconds())
    print("test seconds:", k2.total_seconds())
    print("all seconds:", k3.total_seconds())
#print("max accurancy: ", tf.reduce_max(tf.convert_to_tensor(historyMeanAccurancy)))

#print(historyMeanAccurancy)
#print(batchSize," ", trainingPercen, " ", lr, " ", classNum)
print(dataFilePath)
print(len(trainSamples))
print(len(testSamples))
input()

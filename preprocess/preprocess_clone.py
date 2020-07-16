# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 17:07:00 2019

@author: bzli
"""
import os
from GraphStructure import Graph
import pickle
import random

def loadData(dataFilePath, priorList):
    graphList = []
    labelList = []
    dirs = os.listdir(dataFilePath)
    for classPathName, classIndex in zip(dirs, range(len(dirs))):
        for graphPathName in os.listdir(dataFilePath + classPathName):
            graph = Graph(dataFilePath + classPathName + "/" + graphPathName, priorList, "#")
            graphList.append(graph)
            labelList.append(classIndex)
    return graphList, labelList

def generateCartesianProduct(graphList, labelList, filt = False):        
    samples = []
    labels = []
    for g1,classLable1,index1 in zip(graphList,labelList, range(len(graphList))):
        for g2,classLabel2,index2 in zip(graphList,labelList, range(len(graphList))):
            if index2 <= index1 and filt:
                continue
            samples.append([g1,g2])
            if classLable1 == classLabel2:
                labels.append(1)
            else:
                labels.append(0)
    return samples, labels

def shuffleData(samples, labels):
    seedList = list(range(len(samples)))
    random.shuffle(seedList)
    shuffledSamples = []
    shuffleLabels = []
    for seed in seedList:
        shuffledSamples.append(samples[seed])
        shuffleLabels.append(labels[seed])
    return shuffledSamples, shuffleLabels

def selectSamples(samples, labels, negativeRate):
    shuffleData(samples, labels)
    negativaSampleNum = int(len(samples) * negativeRate)
    positiveSamples = []
    negativeSamples = []
    for sample,label in zip(samples, labels):
        if label == 0 and len(negativeSamples) < negativaSampleNum:
            positiveSamples.append(sample)
            continue
        if label == 1 and len(positiveSamples) < len(samples) - negativaSampleNum:
            negativeSamples.append(sample)
            continue
#    if len(positiveSamples) > len(negativeSamples):
#        positiveSamples = positiveSamples[0:len(negativeSamples)]
#    else:
#        negativeSamples = negativeSamples[0:len(positiveSamples)]
    pl = [1]*len(positiveSamples)
    pl.extend([0]*len(negativeSamples))
    positiveSamples.extend(negativeSamples)
    return positiveSamples, pl

def splitTrainTest(samples, labelList, trainPercent):
    trainSample = []
    trainLabel = []
    testSampel = []
    testLabel = []
    
    for classLabel in range(max(labelList)+1):
        csp = [sam for sam, lab in zip(samples, labelList) if lab == classLabel]
        trainNum = int(len(csp) * trainPercent)
        trainSample.extend(csp[0:trainNum])
        trainLabel.extend([classLabel] * trainNum)
        testSampel.extend(csp[trainNum:])
        testLabel.extend([classLabel] * (len(csp)-trainNum))
        
    return trainSample, trainLabel, testSampel, testLabel

def generateNegativeSamplesList(samples, labels):
    #针对每一种calss生成负样本候选池
    negativePool = {}
    for sample, label in zip(samples, labels):
        if label == 0:
            clsName1 = sample[0].getClassName()
            clsName2 = sample[1].getClassName()
            #print(clsName)
#            if sample[0].getMethodFilePath().find("972839188450312192_数字加密" )>0:
#                print("")
            if negativePool.get(clsName1) is None:
                negativePool[clsName1] = []
                
            negativePool[clsName1].append(sample)
            
            if negativePool.get(clsName2) is None:
                negativePool[clsName2] = []
            negativePool[clsName2].append(sample)
    return negativePool

def dealPath(path):
    #return path
    return "/".join(path.split('/')[-2:])

def generateNegativeSamples(nm, originalSamples, nm_iden, originalSamplesIden):
#    if epo %5 ==0:
#        negMap.clear()
    negaList = []
    negaListIden = []
    i = 0
    for sample, sampleIden in zip(originalSamples, originalSamplesIden):
        print(i)
        assert(dealPath(sample[0].getMethodFilePath())==dealPath(sampleIden[0].getMethodFilePath()))
        clsName = sample[0].getClassName()
        clsNameIden = sampleIden[0].getClassName()
        
        validNegativeSamples = nm[clsName]
        validNegativeSamplesIden = nm_iden[clsNameIden]
        
        validNegativeSamples = [vali for vali in validNegativeSamples if vali[0].getMethodFilePath() == sample[0].getMethodFilePath() or vali[1].getMethodFilePath() == sample[0].getMethodFilePath()]
        validNegativeSamplesIden = [vali for vali in validNegativeSamplesIden if vali[0].getMethodFilePath() == sampleIden[0].getMethodFilePath() or vali[1].getMethodFilePath() == sampleIden[0].getMethodFilePath()]
        
        ind = random.randint(0, len(validNegativeSamples)-1)
        negaList.append(validNegativeSamples[ind])
        negaListIden.append(validNegativeSamplesIden[ind])
        assert(dealPath(validNegativeSamples[ind][1].getMethodFilePath()), dealPath(validNegativeSamplesIden[ind][1].getMethodFilePath()))
        i += 1
    return negaList, negaListIden

def generateNegativeSamples2(nm, originalSamples, nm_iden, originalSamplesIden):
#    if epo %5 ==0:
#        negMap.clear()
    negaList = []
    negaListIden = []
    negMap = {}
    negMapIden={}
    i = 0
    for sample, sampleIden in zip(originalSamples, originalSamplesIden):
        print(i)
        assert(dealPath(sample[0].getMethodFilePath())==dealPath(sampleIden[0].getMethodFilePath()))
        if negMap.get(sample[0].getMethodFilePath()) is not None and negMapIden.get(sampleIden[0].getMethodFilePath()) is not None:
            negaList.append(negMap.get(sample[0].getMethodFilePath()))
            negaListIden.append(negMapIden.get(sampleIden[0].getMethodFilePath()))
            continue
        clsName = sample[0].getClassName()
        clsNameIden = sampleIden[0].getClassName()
        
        validNegativeSamples = nm[clsName]
        validNegativeSamplesIden = nm_iden[clsNameIden]
        
        validNegativeSamples = [vali for vali in validNegativeSamples if vali[0].getMethodFilePath() == sample[0].getMethodFilePath() or vali[1].getMethodFilePath() == sample[0].getMethodFilePath()]
        validNegativeSamplesIden = [vali for vali in validNegativeSamplesIden if vali[0].getMethodFilePath() == sampleIden[0].getMethodFilePath() or vali[1].getMethodFilePath() == sampleIden[0].getMethodFilePath()]
        
        ind = random.randint(0, len(validNegativeSamples)-1)
        negMap[sample[0].getMethodFilePath()] = validNegativeSamples[ind]
        negMapIden[sampleIden[0].getMethodFilePath()] = validNegativeSamplesIden[ind]
        negaList.append(validNegativeSamples[ind])
        negaListIden.append(validNegativeSamplesIden[ind])
        assert(dealPath(validNegativeSamples[ind][1].getMethodFilePath()), dealPath(validNegativeSamplesIden[ind][1].getMethodFilePath()))
        i += 1
    return negaList, negaListIden


priorList = {'assign':-5,"return":-5,"param":-2,"~":6 ,'p++':6,"p--":6,"+":4, "-":4, "++":6,"--":6,"*":5 ,"/":5, "%":5, "!=":1, "==":1,
             "<":2,">":2,"<=":2,">=":2,"&&":-2,"&":0,"||":-3,"|": -1,"c_*":6, "c_+":6,"c_-":6,"^":6,
             "invoke":-4, "parammix":-3, "sizeof":6 ,"<<":3, ">>":3,"!":6, "c_&":6,"c_|":6, "(":7,")":7,"structureaccess":7}

dataFilePath = "../OJ_CLONE_217/"
dataFilePathIden = "../OJ_CLONE_217_IDEN_FILTERED/"

#dataFilePath = "../3_1_test/"
#dataFilePathIden = "../3_1_test/"

graphList, labelList = loadData(dataFilePath, priorList)
trainSample, trainLabel, testSampel, testLabel = splitTrainTest(graphList, labelList, 0.7)
train_samples, train_labels = generateCartesianProduct(trainSample, trainLabel, False)
test_samples, test_labels = generateCartesianProduct(testSampel, testLabel, True)

graphListIden, labelListIden = loadData(dataFilePathIden, priorList)
trainSampleIden, trainLabelIden, testSampelIden, testLabelIden = splitTrainTest(graphListIden, labelListIden, 0.7)
train_samples_iden, train_labels_iden = generateCartesianProduct(trainSampleIden, trainLabelIden, False)
test_samples_iden, test_labels_iden = generateCartesianProduct(testSampelIden, testLabelIden, True)

negativeMap = generateNegativeSamplesList(train_samples, train_labels)
negativeMapIden = generateNegativeSamplesList(train_samples_iden, train_labels_iden)

negaSamples ,negaSamplesIden= generateNegativeSamples2(negativeMap, [sample for sample, label in zip(train_samples,train_labels) if label == 1], negativeMapIden, [sample for sample, label in zip(train_samples_iden,train_labels_iden) if label == 1])

#samples, labels = selectSamples(samples, labels, 0.5)
#os.mkdir("./data")
pickle.dump(train_samples, open("./data/train_samples.pickle",'wb'))
pickle.dump(train_labels, open("./data/train_labels.pickle",'wb'))
#pickle.dump(negativeMap, open("./data_iden/negative_samples.pickle",'wb'))
pickle.dump(test_samples, open("./data/test_samples.pickle",'wb'))
pickle.dump(test_labels, open("./data/test_labels.pickle",'wb'))
pickle.dump(negaSamples, open("./data/nega_samples.pickle",'wb'))

pickle.dump(train_samples_iden, open("./data_iden/train_samples.pickle",'wb'))
pickle.dump(train_labels_iden, open("./data_iden/train_labels.pickle",'wb'))
#pickle.dump(negativeMap, open("./data_iden/negative_samples.pickle",'wb'))
pickle.dump(test_samples_iden, open("./data_iden/test_samples.pickle",'wb'))
pickle.dump(test_labels_iden, open("./data_iden/test_labels.pickle",'wb'))
pickle.dump(negaSamplesIden, open("./data_iden/nega_samples.pickle",'wb'))

print("finish")
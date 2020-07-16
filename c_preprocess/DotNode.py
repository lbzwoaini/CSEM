# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 10:05:05 2019

@author: bzli
"""

class DotNode:
    def __init__(self):
        self.preIds = []
        self.edgeLabels = {}
        self.text = None
        
    def setId(self, idx):
        self.idx = idx
    
    def getId(self):
        return self.idx
    
    def setText(self, text):
        self.text = text
        
    def getText(self):
        return self.text
    
    def setPreIds(self, newIds):
        self.preIds = newIds
        
    def getPreIds(self):
        return self.preIds
    
    def setEdgeLabel(self, preId, label):
        self.edgeLabels[preId] = label
    
    def getEdgeLabels(self):
        return self.edgeLabels
    
    def getLabelByKey(self, key):
        if self.edgeLabels.get(key) is not None:
            return self.edgeLabels.get(key)
        else:
            return ""
    
    def addPreId(self, preId):
        int(preId)
        for idx in self.preIds:
            if preId == idx:
                return
        self.preIds.append(preId)
            
    def addPreIds(self, ids):
        for i in ids:
            self.addPreId(i)

    def setLevel(self, level):
        self.level = level
        
    def getLevel(self):
        return self.level
    
    def setType(self, nodeType):
        self.type = nodeType
        
    def getType(self):
        return self.type
    
    def removePreId(self, idx):
        self.preIds.remove(idx)
        
    @staticmethod
    def mergeNode(node1, node2, intrnalStr):
        node1.setText(node1.getText() + intrnalStr + node2.getText())
        return node1
    
    @staticmethod
    def getDotNode(nodeList, nodeId):
        targetNodes = [node for node in nodeList if node.getId == nodeId]
        if len(targetNodes) == 0:
            return None
        if len(targetNodes) > 1:
            raise Exception("node检出数量超过一个")
        return targetNodes[0]
    
    @staticmethod
    def listAdd(destList, sourceListOrNode):
        if sourceListOrNode == None or len(sourceListOrNode) == 0:
            return
        if destList == None:
            destList = []
        if isinstance(sourceListOrNode, list):
            if len(destList) != 0:
                sourceListOrNode[0].addPreId(destList[-1].getId())
            for node in sourceListOrNode:
                existNodeIds = [ele.getId() for ele in destList]
                if len([cid for cid in existNodeIds if cid == node.getId()]) > 0:
                    continue
                else:
                    destList.append(node)
        else:
            existNodes = [ele for ele in destList if ele.getId() == sourceListOrNode.getId()]
            if len(existNodes) == 0:
                if len(destList) != 0:
                    sourceListOrNode.addPreId(destList[-1].getId())
                destList.append(sourceListOrNode)
              
    def findHouNodes(node, nodeList):
        resultList = []
        for currentNode in nodeList:
            if node.getId() == currentNode.getId():
                continue
            preIds = currentNode.getPreIds()
            for idx in preIds:
                if idx == node.getId():
                    resultList.append(currentNode)
        return resultList
    
    def findNodeByIds(nodeList, nodeIds):
        resultList = []
        for idx in nodeIds:
            for node in nodeList:
                if node.getId() == idx:
                    resultList.append(node)
        if len(resultList) != len(nodeIds):
            raise Exception("数量不等")
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 15:05:59 2019

@author: bzli
"""

from __future__ import print_function
import sys
from pycparser import c_parser, c_ast, parse_file, parse_text
from DotNode import DotNode
import re
import os
import shutil
import re
import time

class IdGenerator:
    def __init__(self, startNum):
        self.startNum = startNum
        self.num = startNum
    
    def nextId(self):
        self.num += 1
        return self.num
    
    def lastId(self):
        return self.num - 1
    
    def currentNum(self):
        return self.num - self.startNum
    
class FuncCallVisitor(c_ast.NodeVisitor):
    def __init__(self, filePath):
        self.filePath = filePath
        self.idGen = IdGenerator(0)
        self.edgeIndex = 1
        self.funcNames = set()
        self.metaVariables = {}
        self.metaMethodNames = {}
        self.idType = None
        self.pathDilema = re.compile("\\|/")
        
    def visit_FuncDef(self, node):
        
        self.funcName = node.decl.name
        #检查是否为重载函数
        index = 0
        while True:
            currentLen = len(self.funcNames)
            self.funcNames.add(self.funcName)
            if len(self.funcNames) == currentLen:
                self.funcName = self.funcName + str(index)
                index += 1
            else:
                break
            
        nodeList = self.makeNodeList(node.body)
        self.filterNodeList(nodeList)
        nodeList = [node for node in nodeList if node.getText() is not None]
        if self.saveAsFile(nodeList, self.filePath + self.funcName + ".txt") == False:
            shutil.rmtree(outFileDir)
            return
        self.saveMetaFile(self.filePath + "meta.data")
        self.saveDotFile(nodeList, self.filePath + self.funcName + ".dot")
        
    def saveMetaFile(self, filePath):
        nameLine = ""
        funcLine = ""
        sortedNameItems = sorted(self.metaVariables.items(), key = lambda x:x[1])
        sortedFuncItems = sorted(self.metaMethodNames.items(), key = lambda x:x[1])
        for it in sortedNameItems:
            nameLine += it[0]+ ":" + str(it[1]) + "\t"
        nameLine += "\n"
        for it in sortedFuncItems:
            funcLine += it[0] + ":" + str(it[1]) + "\t"
        funcLine += "\t"
        f = open(filePath, 'w')
        f.write(nameLine)
        f.write(funcLine)
        f.close()
        
    def saveAsFile(self, nodeList, filePath):
        if len(nodeList) == 0:
            return False
        f = open(filePath, 'w')
        startLine = "node\tnode0\tstart\tstartNode\n"
        f.write(startLine)   
        f.write("edge\tedge"+str(self.edgeIndex)+"\tnode0->node1\t"+""+"\n")
        for node in nodeList:
            f.write(self.generateNodeText(node) + "\n")
            for line in self.generateEdgeText(node):
                f.write(line + "\n")
        f.close()
        return True
    def saveDotFile(self, nodeList, path):
        f = open(path, 'w')
        head = "%s%s%s%s%s%s%s"%("digraph G{\n\tgraph [fontname=\"Microsoft YaHei\"];\n\t",
				"subgraph cluster_g{\n\t\tlabel=<<font color=\"red\">",
				self.funcName,"流程图",
				 "</font>>;\n\t\tnode [shape=record,fontname=\"Microsoft YaHei\"];\n",
				"\t\tedge[fontname=\"Microsoft YaHei\"];\n\n",
				"\t\tnode0[shape=circle,label=\"start\",style=\"filled\",fillcolor=green];\n")
        f.write(head)
        for node in nodeList:
            if node.getType() == "if":
                t = "diamond"
            else:
                t = "\"\""
            
            f.write("\t\tnode%d[label=\"%d:\\n%s\",shape=%s];\n"%(node.getId(), node.getId(), node.getText(), t))
            for preId in node.getPreIds():
                if node.getLabelByKey(preId) is not None or node.getLabelByKey(preId) != "":
                    label = "[label=\"%s\"]"%(node.getLabelByKey(preId))
                else:
                    label = ""
                f.write("\t\tnode%d->node%d%s;\n"%(preId, node.getId(), label))
        f.write("\t}\n}")
        f.close()
            
        
    def generateNodeText(self, node):
        text = "%s\t%s\t%s\t%s"
        lineType = "node"
        lineName = "node" + str(node.getId())
        nodeType = node.getType()
        nodeContent = node.getText()
        return text%(lineType, lineName, nodeType, nodeContent)
    
    def generateEdgeText(self, node):
        resultList = []
        for preId in node.getPreIds():
            text = "%s\t%s\t%s\t%s"
            lineType = "edge"
            lineName = "edge" + str(self.edgeIndex)
            self.edgeIndex += 1
            edgeType = "node" + str(preId) + "->" + "node" + str(node.getId())
            #print(edgeType)
            edgeContent = node.getLabelByKey(preId)
            resultList.append(text%(lineType, lineName, edgeType, edgeContent))
        return resultList

    def filterNodeList(self, nodeList):
        #过滤空的入口和出口节点
        for node in nodeList:
            if node.getText() is None:
                preIds = node.getPreIds()
                houNodes = DotNode.findHouNodes(node, nodeList)
                #删除空节点并连接节点前后关系
                for n in houNodes:
                    n.removePreId(node.getId())
                    n.addPreIds(preIds)
                    els = node.getEdgeLabels()
                    for pi, ps in els.items():
                        n.setEdgeLabel(pi, ps)
                    
    #处理一级结构: if, switch, while, do-while, for
    def makeNodeList(self, astNode):
        ###若非上述结构，直接转发至dealExpression
        ###
        #astNode.show()
        if astNode == None:
            return None
        nodeList = []
        if isinstance(astNode, c_ast.Compound):
            statementList = astNode.block_items
            if statementList == None:
                return []
            for statement in statementList:
                DotNode.listAdd(nodeList, self.makeNodeList(statement))
            return nodeList
        
        if isinstance(astNode, c_ast.If):
            #获取if语句条件表达式调用链
            condNode = self.makeNodeList(astNode.cond)[0]
            condNode.setType("if")
            nodeList.append(condNode)
            iftrue = astNode.iftrue
            iffalse = astNode.iffalse
            
            #处理真条件方法体
            if iftrue != None:
                iftrueNodeList = self.makeNodeList(iftrue)
                #连接cond节点与iftrue子结构
                iftrueNodeList[0].addPreId(condNode.getId())
                iftrueNodeList[0].setEdgeLabel(condNode.getId(), "yes")
                nodeList.extend(iftrueNodeList)
            
            #处理条件为假的情况
            if iffalse != None:
                iffalseNodeList = self.makeNodeList(iffalse)
                #连接cond节点与iftrue子结构
                iffalseNodeList[0].addPreId(condNode.getId())
                iffalseNodeList[0].setEdgeLabel(condNode.getId(), "no")
                nodeList.extend(iffalseNodeList)

            #构造出口空节点
            outNode = DotNode()
            outNode.setId(self.idGen.nextId())
            if iftrue != None:
                outNode.addPreId(iftrueNodeList[-1].getId())
            if iffalse != None:    
                outNode.addPreId(iffalseNodeList[-1].getId())
            else:
                outNode.addPreId(condNode.getId())
                outNode.setEdgeLabel(condNode.getId(), "no")
            nodeList.append(outNode)
            return nodeList
        
        if isinstance(astNode, c_ast.While):
            condNode = self.makeNodeList(astNode.cond)[0]
            condNode.setType("if")
            #condNode.setType("whille")
            nodeList.append(condNode)
            stmtNodeList = self.makeNodeList(astNode.stmt)
            stmtNodeList[0].addPreId(condNode.getId())
            stmtNodeList[0].setEdgeLabel(condNode.getId(), "yes")
            nodeList.extend(stmtNodeList)
            #构造循环边
            condNode.addPreId(stmtNodeList[0].getId())
            
            #构造出口空节点
            outNode = DotNode()
            outNode.setId(self.idGen.nextId())
            #outNode.addPreId(condNode.getId())
            outNode.setEdgeLabel(condNode.getId(), "no")
            #outNode.addPreId(stmtNodeList[-1].getId())
            nodeList.append(outNode)
            return nodeList
        
        if isinstance(astNode, c_ast.For):
            t = self.makeNodeList(astNode.cond)
            if t == None:
                #print("None")
                condNode = DotNode()
                condNode.setId(self.idGen.nextId())
                condNode.setText("constantBool")
            else:
                condNode = t[0]
            condNode.setType("if")
            #condNode.setType("For")
            nodeList.append(condNode)
            stmtNodeList = self.makeNodeList(astNode.stmt)
            stmtNodeList[0].addPreId(condNode.getId())
            stmtNodeList[0].setEdgeLabel(condNode.getId(), "yes")
            nodeList.extend(stmtNodeList)
            #构造循环边
            condNode.addPreId(stmtNodeList[0].getId())
            
            #构造出口空节点
            outNode = DotNode()
            outNode.setId(self.idGen.nextId())
            #outNode.addPreId(condNode.getId())
            outNode.setEdgeLabel(condNode.getId(), "no")
            outNode.addPreId(condNode.getId())
            #outNode.addPreId(stmtNodeList[-1].getId())
            nodeList.append(outNode)
            return nodeList
        
        if isinstance(astNode, c_ast.DoWhile):
            stmtNodeList = self.makeNodeList(astNode.stmt)
            condNode = self.makeNodeList(astNode.cond)[0]
            condNode.setType("if")
            condNode.addPreId(stmtNodeList[-1].getId())
            stmtNodeList[0].addPreId(condNode.getId())
            stmtNodeList[0].setEdgeLabel(condNode.getId(), "yes")
            nodeList.extend(stmtNodeList)
            nodeList.append(condNode)
            #构造do-while出口节点
            outNode = DotNode()
            outNode.setId(self.idGen.nextId())
            outNode.addPreId(condNode.getId())
            outNode.setEdgeLabel(condNode.getId(), "no")
            nodeList.append(outNode)
            return nodeList
        
        #switch结构直接转为多个if结构
        if isinstance(astNode, c_ast.Switch):
            cases = astNode.stmt.block_items
            caseNodeLists = []
            for case in cases:
                condNodes = self.makeNodeList(astNode.cond)
                caseNodeList = self.makeNodeList(case)
                DotNode.mergeNode(condNodes[0], caseNodeList[0], "#==#")
                idx = condNodes[0].getId()
                condNodes[0].setType("if")
                caseNodeList[1].removePreId(caseNodeList[0].getId())
                caseNodeList[1].addPreId(idx)
                condNodes.extend(caseNodeList[1:])
                caseNodeLists.append(condNodes)
            #添加入口和出口节点
            inputNode = DotNode()
            inputNode.setId(self.idGen.nextId())
            for caseNodeList in caseNodeLists:
                caseNodeList[0].addPreId(inputNode.getId())
                
            #添加出口节点
            outNode = DotNode()
            outNode.setId(self.idGen.nextId())
            outNode.addPreIds([i[0].getId() for i in caseNodeLists])
            
            #构造结构整体
            nodeList.append(inputNode)
            for caseNodeList in caseNodeLists:
                nodeList.extend(caseNodeList)
            nodeList.append(outNode)
            
            return nodeList
        
        if isinstance(astNode, c_ast.Case):
            condNode = self.makeNodeList(astNode.expr)[0]
            stms = astNode.stmts
            stmsList = []
            for stm in stms:
                if isinstance(stm, c_ast.Break):
                    continue
                contentNodeList = self.makeNodeList(stm)
                stmsList.append(contentNodeList[0])
                
            stmsList[0].addPreId(condNode.getId())
            nodeList.append(condNode)
            nodeList.extend(stmsList)
            return nodeList

        if isinstance(astNode, c_ast.Default):
            condNode = DotNode()
            condNode.setId(self.idGen.nextId())
            condNode.setText("default")
            stms = astNode.stmts
            stmsList = []
            for stm in stms:
                contentNodeList = self.makeNodeList(stm)
                stmsList.append(contentNodeList[0])
                
            stmsList[0].addPreId(condNode.getId())
            nodeList.append(condNode)
            nodeList.extend(stmsList)
            return nodeList
        if isinstance(astNode, c_ast.ExprList):
            exprs = astNode.exprs
            for expr in exprs:
                exprList1 = self.makeNodeList(expr)
                if len(nodeList) != 0:
                    exprList1[0].addPreId(nodeList[-1].getId())
                nodeList.extend(exprList1)
            return nodeList
        if isinstance(astNode, c_ast.EmptyStatement):
            outNode = DotNode()
            outNode.setId(self.idGen.nextId())
            return [outNode]
        #非指定结构的处理方式
        exList = self.dealExpression(astNode)
        dotNode = DotNode()
        dotNode.setType("expression")
        dotNode.setText("".join(exList))
        dotNode.setId(self.idGen.nextId())
        nodeList.append(dotNode)
        return nodeList
    
    #expression是比statement小的单元，一个statement可能包含多个expression
    def dealExpression(self, expression):
        #expList.append(str(type(expression)))
        if expression == None:
            return ["null"]
        
        if isinstance(expression, c_ast.BinaryOp):
            op = expression.op
            left = expression.left
            right = expression.right
            leftStack = self.dealExpression(left)
            rightStack = self.dealExpression(right)
            leftStack.insert(0, "#(#")
            leftStack.append("#)#")
            leftStack.extend("#" + op + "#")
            rightStack.insert(0, "#(#")
            rightStack.append("#)#")
            leftStack.extend(rightStack)
            return leftStack
        
        if isinstance(expression, c_ast.ID):
            if self.idType == "funcName":
                metaMap = self.metaMethodNames
            else:
                metaMap = self.metaVariables
                
            if metaMap.get(expression.name) is None:
                metaMap[expression.name] = 0
            else:
                metaMap[expression.name] += 1
            return [expression.name]
        
        if isinstance(expression, c_ast.Return):
            expr = expression.expr
            if expr is None:
                return ["#return#", "None"]
            else:
                exl = self.dealExpression(expr)
                result = ["#return#"]
                result.extend(exl)
                return result
        
        if isinstance(expression, c_ast.Constant):
            expType = expression.type
            if expType == 'char':
                return ["constantStr"]
            if expType == "string":
                return ["constantStr"]
            else:
                return ["constantNum"]
        
        if isinstance(expression, c_ast.Decl):
            init = expression.init
            name = expression.name
            initList = self.dealExpression(init)
            result = [name,"#assign#"]
            result.extend(initList)
            return result
        
        if isinstance(expression, c_ast.Assignment):
            lvalue = expression.lvalue
            rvalue = expression.rvalue
            lRes = self.dealExpression(lvalue)
            rRes = self.dealExpression(rvalue)
            resultList = []
            resultList.extend(lRes)
            resultList.append("#assign#")
            resultList.extend(rRes)
            return resultList
        
        if isinstance(expression, c_ast.FuncCall):
            if expression.args != None:
                args = expression.args.exprs
            else:
                args = []
            funcName = expression.name
            self.idType = "funcName"
            funcNameList = self.dealExpression(funcName)
            self.idType = "name"
            argAllList = ['#param##(#']
            for arg, index in zip(args, range(len(args))):
                argList = ["#(#"]
                argList.extend(self.dealExpression(arg))
                argList.append("#)#")
                if index != len(args) and index != 0:
                    argAllList.append("#parammix#")
                argAllList.extend(argList)
            if len(args) == 0:
                argAllList.append("null")
            argAllList.append("#)#")
            resultList = []
            resultList.append("#invoke#")
            resultList.extend(funcNameList)
            resultList.extend(argAllList)
            return resultList
        
        if isinstance(expression, c_ast.UnaryOp):
            op = expression.op
            expr = expression.expr
            exprList = self.dealExpression(expr)
            resList = []
            
            if op == "&" or op =="|" or op == "-" or op == "*" or op=="+":
                resList.append("#c_" + op + "#")
            else:
                resList.append("#" + op + "#")
            resList.append("#(#")
            resList.extend(exprList)
            resList.append("#)#")
            return resList
        if isinstance(expression, c_ast.ArrayRef):
            return ['arrayRef']
        
        if isinstance(expression, c_ast.InitList):
            return ["constantArray"]
        
        if isinstance(expression, c_ast.Break):
            return ["#invoke#break"]
        if isinstance(expression, c_ast.StructRef):
            nameList = self.dealExpression(expression.name)
            fieldList = self.dealExpression(expression.field)
            nameList.insert(0, "#(#")
            nameList.append("#structureaccess#")
            nameList.extend(fieldList)
            nameList.append("#)#")
            return nameList
        return ["unknown"]
def dealFile(inputFilePath, outputDir):
    ast = parse_file(inputFilePath)#, use_cpp=True, cpp_path = "gcc", cpp_args=['-E', r'-Iutils/fake_libc_include'])
    v = FuncCallVisitor(outputDir)
    v.visit(ast)
    
def dealFileByText(inputFileText, outputDir):
    #print("input ast")
    ast = parse_text(inputFileText)
    v = FuncCallVisitor(outputDir)
    v.visit(ast)
    #print("output ast")
    
def generateTempFile(sourceFilePath, outfileDir):
    if not os.path.exists(outfileDir):
        os.makedirs(outfileDir)
    f = open(sourceFilePath,'r',encoding = "utf-8")
    lines = f.readlines()
    f.close()
    newLines = []
    for line in lines:
        if line.find("#include") >= 0 or line.find("#define") >= 0:
            continue
#        if line.strip().startswith("//"):
#            continue
        if line.strip().startswith("using"):
            continue
        else:
            newLines.append(line)
    tempStr = "".join(newLines)
    tempStr = re.sub(r"\/\*([^\*^\/]*|[\*^\/*]*|[^\**\/]*)*\*\/", "",tempStr)
    tempStr = re.sub(r"\/\/[^\n]*", "", tempStr)
    of = open(outfileDir + sourceFilePath.split("/")[-1], 'w')
    of.write(tempStr)
    of.close()
    return outfileDir + sourceFilePath.split("/")[-1]

def generateTempStr(sourceFilePath):
    f = open(sourceFilePath,'r',encoding = "utf-8")
    lines = f.readlines()
    f.close()
    newLines = []
    for line in lines:
        if line.find("#include") >= 0 or line.find("#define") >= 0:
            continue
#        if line.strip().startswith("//"):
#            continue
        if line.strip().startswith("using"):
            continue
        else:
            newLines.append(line)
    tempStr = "".join(newLines)
    #tempStr = re.sub(r"\/\*(\s|.)*?\*\/", "",tempStr)
    tempStr = re.sub(r"\/\/[^\n]*", "", tempStr)
    return tempStr
    


expList = []
valid = 0
invalid = 0
perClassNum = 300
tempOutDir = "../temp_out/"
if __name__ == "__main__":
    inputFilePath = "../OJ_CLONE_217_ORIGIN/"
    outFilePath = "../OJ_CLONE_217/"
    for questionName in os.listdir(inputFilePath):
        ccNum = 0
        questionDir = inputFilePath + questionName + "/"
        for codeFileName in os.listdir(questionDir):
            if ccNum > perClassNum:
                    break
            if codeFileName.split(".")[-1] != "txt":
                continue
            codeFilePath = questionDir + codeFileName
            outFileDir = outFilePath + questionName + "/" + codeFileName + "/"
            try:
                if not os.path.exists(outFileDir):
                    os.makedirs(outFileDir)
                #print(codeFilePath)

                #dealFile(generateTempFile(codeFilePath, tempOutDir), outFileDir)
                #dealFile(codeFilePath, outFileDir)
                
                dealFileByText(generateTempStr(codeFilePath), outFileDir)
                valid += 1
                ccNum += 1
            except Exception as e:
                try:
                    shutil.rmtree(outFileDir)
                except Exception as ee:
                    print(outFileDir)
                invalid += 1
                #print(e)
print(valid)
print(invalid)       

#if __name__ == "__main__":
#    inputFilePath = "../题目分组/"
#    outFilePath = "../1121/"
#    dealFile(generateTempFile(r"E:/program_embedding_literature/ProgramData/26/803.txt", tempOutDir), "../test_out_dir/")
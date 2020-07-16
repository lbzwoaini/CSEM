# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 12:19:45 2019

@author: bzli
"""
#模板文件,需要保存为csv格式
replaceDictFilePath = r"C:\Users\bzli\Desktop\replace\模板.csv"

#需要进行替换的源文件
sourceFilePath = r"C:\Users\bzli\Desktop\replace\source_file.txt"

#生成的经过替换的文件路径
outFilePath = r"C:\Users\bzli\Desktop\replace\out_file.txt"

#构造替换Map
csvFile = open(replaceDictFilePath, 'r')
replaceMap = {}
for line in csvFile:
    line = line.strip()
    if line == '':
        continue
    temp = line.split(",")
    replaceMap[temp[0].strip()] = temp[1].strip()
    
#读取需要被替换的文件
sourceFile = open(sourceFilePath, 'r')
newFileLines = []
for line in sourceFile:
    #针对被替换文件的每一行应用所有替换规则
    tempLine = line.strip()
    print("tempLine:", tempLine)
    for key in replaceMap.keys():
        print(replaceMap[key])
        tempLine = tempLine.replace(key, replaceMap[key])
        print("replacedTempLine:", tempLine)
    newFileLines.append(tempLine + "\n")
    
#将新生成的文件写入指定目录
f = open(outFilePath, "w")
for line in newFileLines:
    f.write(line)
f.close()
print("替换完成")
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 11:02:50 2020

@author: bzli
"""
import pickle
import shutil
import os

original_path = "E:/program_embedding_literature/OJ_CLONE_217_ORIGIN/"
dest_train_path = "E:/program_embedding_literature/OJ_CLONE_217_FILTERED/"
dest_test_path = "E:/program_embedding_literature/OJ_CLONE_217_FILTERED/"
test_pickle_file_path = "E:/program_embedding_literature/code_clone/data/test_samples.pickle"
train_pickle_file_path = "E:/program_embedding_literature/code_clone/data/train_samples.pickle"

def get_pickle_object(path):
    return pickle.load(open(path, 'rb'))

def generate_file_path_set(pickleFile):
    file_path_list = []
    for element in pickleFile:
        file_path_list.append(element[0].getMethodFilePath())
        file_path_list.append(element[1].getMethodFilePath())
    return set(file_path_list)

def generate_copy_paths(source, dest, target):
    subfix = '/'.join(target.split("/")[-2:])
    return source + subfix, dest + subfix

def copy_file(source, dest):
    dest_parent_path = "/".join(dest.split('/')[0:-1])
    if not os.path.exists(dest_parent_path):
        os.makedirs(dest_parent_path)
    shutil.copy(source, dest)
    
def deal_one(pickle_path, ori_path, dest_path):
    pf = get_pickle_object(pickle_path)
    paths = generate_file_path_set(pf)
    for p in paths:
        source, dest = generate_copy_paths(ori_path, dest_path, p)
        copy_file(source, dest)
        
def main():
    deal_one(train_pickle_file_path, original_path, dest_train_path)
    deal_one(test_pickle_file_path, original_path, dest_test_path)
    
main()
import os
import sys
import re                                                                                                                  
import shutil                                                                                                              
import argparse
import yaml
from collections import OrderedDict                                                                                        
from datetime import datetime                                                                                              
from glob import glob  

import torch

def make_dirs(*paths):                                                                                                     
    os.makedirs(path(*paths), exist_ok=True)                                                                               
                                                                                                                           
                                                                                                                           
def makedir(dir):                                                                                                          
    if not os.path.exists(dir):                                                                                            
        os.makedirs(dir)                                                                                                   
                                                                                                                           
                                                                                                                           
def _parsing():                                                                                                            
    parser = argparse.ArgumentParser()                                                                                     
    parser.add_argument('--yaml', type=str, required=True, help='yaml file')                                               
    args = parser.parse_args()                                                                                             
    return args                                                                                                            
                                                                                                                           
                                                                                                                           
def _parsing_opt():                                                                                                        
    parser = argparse.ArgumentParser()                                                                                     
    parser.add_argument('--yaml', type=str, required=True, help='yaml file')                                               
    parser.add_argument('--opt', type=str, required=True, help='yaml opt file')                                            
    args = parser.parse_args()                                                                                             
    return args                                                                                                            
                 

def _ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):                                              
    """                                                                                                                    
        Load parameters from yaml in order                                                                                 
    """                                                                                                                    
                                                                                                                           
    class OrderedLoader(Loader):                                                                                           
        pass                                                                                                               
                                                                                                                           
    def construct_mapping(loader, node):                                                                                   
        loader.flatten_mapping(node)                                                                                       
        return object_pairs_hook(loader.construct_pairs(node))                                                             
                                                                                                                           
    OrderedLoader.add_constructor(                                                                                         
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,                                                                    
        construct_mapping)                                                                                                 
                                                                                                                           
    # print(dict(yaml.load(stream, OrderedLoader).items()))                                                                
                                                                                                                           
    return yaml.load(stream, OrderedLoader)                                                                                
                                                                                                                           
                                                                                                                           
def _print_config(config, config_path):                                                                                    
    """Print config in dictionary format"""                                                                                
    print("\n====================================================================\n")                                      
    print('RUNNING CONFIG: ', config_path)                                                                                 
    print('TIME: ', datetime.now())                                                                                        
                                                                                                                           
    for key, value in config.items():                                                                                      
        print(key, value)                                                                                                  
                                                                                                                           
    return 


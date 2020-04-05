# encoding=utf-8
import jieba

"""
Reference: https://github.com/fxsjy/jieba
"""    

def divide(str, mode='default'):
    """
    Input: a setence string
    Mode:
        default: 
            Dividing sentence into the most accurate segmentations without overlap.
        full: 
            Finding all possible words from the sentence. 
        search: 
            Based on default mode, cutting long words into several short words.

    Ouput: list of words segmentations
    """    
    if mode == 'default': 
        seg_list = jieba.lcut(str)
        return seg_list
    elif mode == 'full': 
        seg_list = jieba.lcut(str,cut_all=True)
        return seg_list
    elif mode == 'search':
        seg_list = jieba.lcut_for_search(str)
        return seg_list


def print_seg_list(seg_list):
    """
    Print out the word segmentations list with format : {} | {} | {}
    """    
    print(" | ".join(seg_list))  

def tokenize(str):
    """
    Tokenize sentence string by removing empty space and EOL
    """   
    tokenized_str = str.replace('\n','')
    tokenized_str = tokenized_str.replace('　','')
    tokenized_str = ''.join(tokenized_str.split())
    return tokenized_str

def example():
    """
    Example
    """  
    str = "床前明月光，疑是地上霜。\n举头望明月，低头思故乡。"
    str = tokenize(str)
    print("default:")
    print_seg_list(divide(str))
    print("full:")
    print_seg_list(divide(str,'full'))
    print("search:")
    print_seg_list(divide(str,'search'))

# example()
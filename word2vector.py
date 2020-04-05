# encoding=utf-8
import numpy as np

"""
Reference: https://github.com/Embedding/Chinese-Word-Vectors
"""

class word2vector:
  
  def __init__(self):
    self.word2vectors = dict() 
    self.vector_size = 300

  def parse_word2vector(self, data_file):
    """
    Input: word2vector data file

    Parse word2vector data into the dictionary - {character, vector}
    """
    self.word2vectors = dict()
    with open(data_file) as f:
      line = f.readline()
      # num = int(line.split()[0])
      self.vector_size = int(line.split()[1])
      while line:
        line = f.readline()
        if not line:
          break
        line_split = line.split()
        char = line_split[0]
        vector = []
        for i in range(len(line_split)):
          if i == 0:
            continue
          vector.append(float(line_split[i]))
          self.word2vectors[char] = vector

  def get_vector(self, word):
    """
    Input: a word string
    Output: word vector of dimension(1,300)
            return vector of zero if the word does not any character in the word2vector dict
    """
    if word in self.word2vectors:
      return self.word2vectors[word]
    else:
      char_vectors = []
      for char in word:       
        if char in self.word2vectors:
          char_vectors.append(self.word2vectors[char])
      if len(char_vectors) == 0:
        zeros = [0] * self.vector_size
        char_vectors.append(zeros)
      data = np.array(char_vectors)
      data = np.average(data, axis=0)
      return data.tolist()
      
""" 
Example:
  word2vector = word2vector()
  word2vector.parse_word2vector('sgns.sikuquanshu.bigram')
  print(word2vector.get_vector('李白'))
"""




import json
import math
import random
import os
import sys
import codecs
import json

#file_path = "C:\\Users\80482\Downloads\poetry-master\data\poetry"
#training_fold_list = os.listdir(file_path)
file = open("metadata.txt")
metadata = json.load(file)
file.close()
data = {k:[] for k in metadata["dynasty_set"]}
file = open("gushiwen1.json",encoding="utf-8")
text1 = json.load(file)
file.close()
file = open("gushiwen2.json",encoding="utf-8")
text2 = json.load(file)
file.close()
text = text1 + text2

label_encoding = {}
word_encoding = {}
for i in range(len(metadata["dynasty_set"])):
    label_encoding[metadata["dynasty_set"][i]] = i
for i in range(len(metadata["word_set"])):
    word_encoding[metadata["word_set"][i]] = i + 1



for i in text:
    #file = open(file_path+"\\"+i,encoding="utf-8")
    #item = json.load(file)
    #file.close()
    #content = i["content"].replace("\n", "")+"\n"
    num_string = ""
    count = 0
    for j in i["content"]:
        if count == 100:
            break
        count = count + 1
        num_string = num_string + str(word_encoding[j]) +" "
    data[i["dynasty"]].append(num_string[:-1]+"\n")
print(len(data))
for i in metadata["dynasty_set"]:
    file = open(i+".txt","w",encoding="utf-8")
    file.writelines(data[i])
    file.close()
#output_file = codecs.open("gushiwen1.json","w",encoding="utf-8")
#json.dump(data[:35000],output_file,ensure_ascii=False)
#output_file.close()
#output_file = codecs.open("gushiwen2.json","w",encoding="utf-8")
#json.dump(data[35000:],output_file,ensure_ascii=False)
#output_file.close()


import json
import math
import random

tolerance = 2
test_set_divisor = 10

input_path = "shici.json"
input_file = open(input_path, encoding="utf-8")
article_list = json.load(input_file)
input_file.close()

input_lines = [(x["dynasty"], x["content"]) for x in article_list]
random.shuffle(input_lines)
training_sample = input_lines[:len(input_lines)-len(input_lines)//test_set_divisor]
test_sample = input_lines[len(input_lines)-len(input_lines)//test_set_divisor:]

dynasty_set = set()
for i in input_lines:
    if i[0] not in dynasty_set:
        dynasty_set.add(i[0])
#Calculate Dynasty Prior        
dynasty_count = dict.fromkeys(list(dynasty_set), 0)
for i in training_sample:
    dynasty_count[i[0]] = dynasty_count[i[0]] + 1
dynasty_count["total_count"] = len(training_sample)
dynasty_prior = {}
for dynasty in dynasty_set:
    dynasty_prior[dynasty] = dynasty_count[dynasty] / dynasty_count["total_count"]

#Calculate Conditional Probability for Word
word_set = set()
for i in input_lines:
    for j in i[1]:
        if j not in word_set:
            word_set.add(j)
word_count = dict.fromkeys(list(dynasty_set),dict.fromkeys(list(word_set), 1))
for i in dynasty_set:
    word_count[i]["total_count"] = len(word_set)
for i in training_sample:
    for j in i[1]:
        word_count[i[0]][j] = word_count[i[0]][j] + 1
        word_count[i[0]]["total_count"] = word_count[i[0]]["total_count"] + 1
word_prob = dict.fromkeys(list(dynasty_set), dict.fromkeys(list(word_set), 0))
for i in dynasty_set:
    for j in word_set:
        word_prob[i][j] = word_count[i][j] / word_count[i]["total_count"]

#Write Model Parameters
model_path = "naive_bayes_model.txt"
model_file = open(model_path, "w")
json.dump({"dynasty_prior": dynasty_prior, "word_prob": word_prob}, model_file)
model_file.close()

#Prediction
correct_count = 0
for i in training_sample:
    dyn_prob = {k:-math.log(v) for k,v in dynasty_prior.items()}
    for j in i[1]:
        for k in dynasty_set:
            dyn_prob[k] = dyn_prob[k] - math.log(word_prob[k][j])
    dyn_prob = [k for k,v in sorted(dyn_prob.items(),key= lambda x:x[1])]
    if i[0] in dyn_prob[:tolerance]:
        correct_count = correct_count + 1
print("training_accuracy: " + str(correct_count/len(training_sample)))
correct_count = 0
for i in test_sample:
    dyn_prob = {k:-math.log(v) for k,v in dynasty_prior.items()}
    for j in i[1]:
        for k in dynasty_set:
            dyn_prob[k] = dyn_prob[k] - math.log(word_prob[k][j])
    dyn_prob = [k for k,v in sorted(dyn_prob.items(),key= lambda x:x[1])]
    if i[0] in dyn_prob[:tolerance]:
        correct_count = correct_count + 1
print("test_accuracy: " + str(correct_count/len(test_sample)))

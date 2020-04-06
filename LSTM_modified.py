import tensorflow as tf
import json
import numpy as np
import word_divider
import word2vector

print("Tensorflow Version: "+tf.version.VERSION)
meta_path = "metadata.txt"
meta_file = open(meta_path)
metadata = json.load(meta_file)
meta_file.close()
length_truncation = 100
word_embedding = word2vector.word2vector()
word_embedding.parse_word2vector("word2vector_bigram")

vocab_size = len(metadata["word_set"]) + 1
output_class = len(metadata["dynasty_set"])
label_encoding = {}
for i in range(len(metadata["dynasty_set"])):
    label_encoding[metadata["dynasty_set"][i]] = i
#def encode(text_tensor, label):
#    char_list = []
#    for i in str(text_tensor.numpy())[2:-1].split(" "):
#        char_list.append(int(i))
#    return char_list,label


def labeler(example, index):
    label = tf.cast(index, tf.int64)

    return example, label


def encode(example,label):
    label.set_shape([])
    example.set_shape([length_truncation, 300])
    return example,label

labeled_data_sets = []
#for dynasty in metadata["dynasty_set"]:
#    line_dataset = tf.data.TextLineDataset(dynasty+".txt")
#    labeled_dataset = line_dataset.map(lambda x:labeler(x, label_encoding[dynasty]))
#    labeled_data_sets.append(labeled_dataset)

for dynasty in ["金朝","隋代","魏晋"]:#metadata["dynasty_set"]:
    file = open(dynasty+".json",encoding="utf-8")
    samples = [i["content"] for i in json.load(file)]
    file.close()
    print(dynasty+" Working")
    sets = []
    for i in samples:
        token_list = word_divider.divide(i)
        vector_list = []

        for j in token_list:
            vector_list.append(word_embedding.get_vector(j))
            if len(vector_list) == length_truncation:
                break
        if len(token_list)<length_truncation:
            for _ in range(length_truncation-len(token_list)):
                vector_list.append(np.zeros(shape = [300]))
        sets.append(tf.convert_to_tensor(np.array(vector_list)))#, tf.cast(label_encoding[dynasty],tf.int32)))
    dataset = tf.data.Dataset.from_tensor_slices(sets)
    labeled_dataset = dataset.map(lambda x:labeler(x,label_encoding[dynasty]))
    labeled_data_sets.append(labeled_dataset)
    print(dynasty+" Parsed")
#dataset = tf.data.Dataset.from_tensor_slices(labeled_data_sets)
print(labeled_data_sets)






BUFFER_SIZE = 50
BATCH_SIZE = 5
TAKE_SIZE = 10

all_labeled_data = labeled_data_sets[0]
for labeled_dataset in labeled_data_sets[1:]:
    all_labeled_data = all_labeled_data.concatenate(labeled_dataset)
#print(all_labeled_data)
all_labeled_data = all_labeled_data.shuffle(
    BUFFER_SIZE, reshuffle_each_iteration=True)

#for ex in all_labeled_data.take(5):
#  print(ex)

#def encode_map_fn(text, label):
#    # py_func doesn't set the shape of the returned tensors.
#    encoded_text, label = tf.py_function(encode,
#                                       inp=[text, label],
#                                       Tout=(tf.int64, tf.int64))

    # `tf.data.Datasets` work best if all components have a shape set
    #  so set the shapes manually:
#    encoded_text.set_shape([None])
#    label.set_shape([])

#    return encoded_text, label


all_encoded_data = all_labeled_data.map(encode)
train_data = all_labeled_data.skip(TAKE_SIZE).shuffle(BUFFER_SIZE)
train_data = train_data.batch(BATCH_SIZE)
#print(train_data)
test_data = all_labeled_data.take(TAKE_SIZE)
test_data = test_data.batch(BATCH_SIZE)
sample_text, sample_labels = next(iter(test_data))

#print(sample_text[0])
#print(sample_labels[0])


model = tf.keras.Sequential([
#    tf.keras.layers.Embedding(vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(output_class)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_data, epochs=3, validation_data=test_data)

eval_loss, eval_acc = model.evaluate(test_data)
print('\nEval loss: {}, Eval accuracy: {}'.format(eval_loss, eval_acc))
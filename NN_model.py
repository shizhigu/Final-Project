import tensorflow as tf
import json
print("Tensorflow Version: "+tf.version.VERSION)
meta_path = "metadata.txt"
meta_file = open(meta_path)
metadata = json.load(meta_file)
meta_file.close()

vocab_size = len(metadata["word_set"]) + 1
output_class = len(metadata["dynasty_set"])
label_encoding = {}
word_encoding = {}
for i in range(len(metadata["dynasty_set"])):
    label_encoding[metadata["dynasty_set"][i]] = i
for i in range(len(metadata["word_set"])):
    word_encoding[metadata["word_set"][i]] = i + 1




def encode(text_tensor, label):
    char_list = []
    for i in str(text_tensor.numpy())[2:-1].split(" "):
        char_list.append(int(i))
    return char_list,label


def labeler(example, index):
  return example, tf.cast(index, tf.int64)


labeled_data_sets = []
for dynasty in metadata["dynasty_set"]:
    line_dataset = tf.data.TextLineDataset(dynasty+".txt")
    labeled_dataset = line_dataset.map(lambda x:labeler(x, label_encoding[dynasty]))
    labeled_data_sets.append(labeled_dataset)

BUFFER_SIZE = 50000
BATCH_SIZE = 64
TAKE_SIZE = 5000

all_labeled_data = labeled_data_sets[0]
for labeled_dataset in labeled_data_sets[1:]:
    all_labeled_data = all_labeled_data.concatenate(labeled_dataset)

all_labeled_data = all_labeled_data.shuffle(
    BUFFER_SIZE, reshuffle_each_iteration=True)

def encode_map_fn(text, label):
  # py_func doesn't set the shape of the returned tensors.
  encoded_text, label = tf.py_function(encode,
                                       inp=[text, label],
                                       Tout=(tf.int64, tf.int64))

  # `tf.data.Datasets` work best if all components have a shape set
  #  so set the shapes manually:
  encoded_text.set_shape([None])
  label.set_shape([])

  return encoded_text, label


all_encoded_data = all_labeled_data.map(encode_map_fn)
train_data = all_encoded_data.skip(TAKE_SIZE).shuffle(BUFFER_SIZE)
train_data = train_data.padded_batch(BATCH_SIZE, padded_shapes=([None],[]))

test_data = all_encoded_data.take(TAKE_SIZE)
test_data = test_data.padded_batch(BATCH_SIZE, padded_shapes=([None],[]))
sample_text, sample_labels = next(iter(test_data))

#print(sample_text[0])
#print(sample_labels[0])


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(output_class)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_data, epochs=10, validation_data=test_data)

eval_loss, eval_acc = model.evaluate(test_data)
print('\nEval loss: {}, Eval accuracy: {}'.format(eval_loss, eval_acc))
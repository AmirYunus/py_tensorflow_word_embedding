import collections
import os
import random
import urllib
import zipfile
import numpy as np
import tensorflow as tf

# Training parameters
tf.enable_eager_execution()
learning_rate = 0.1
batch_size =128
number_of_steps = 3_000_000
display_step = 10_000
evaluation_step = 200_000
device = "/cpu:0" # Enforces computation done on CPU
optimiser = tf.keras.optimizers.SGD(learning_rate # Stochastic gradient descent optimiser

# Evaluation parameters
evaluate_words = ["five", "of", "going", "american", "british"]

# Word2Vec parameters
embedding_size = 200 # Dimension of the embedding vector
maximum_vocabulary_size = 50_000 # Total number of different words in the vocabulary
minimum_occurrence = 10 # Remove all words that does not appears at least this number of times
skip_window = 3 # How many words to consider left and right
number_of_skips = 2 # How many times to reuse an input to generate a label
number_of_samples = 64 # Number of negative examples to sample

# Download a small batch of Wikipedia articles collection
url = "https://github.com/AmirYunus/py_tensorflow_word_embedding/blob/master/text.zip"
data_path = "text.zip"

if not os.path.exists(data_path):
    print("Downloading the dataset... (It may take some time)")
    filename, _ = urllib.request.urlretrieve(url, data_path)
    print("Done!")

with zipfile.ZipFile(data_path) as f: # Unzip the dataset file. Text has already been processed
    text_words = f.read(f.namelist()[0]).lower().split()

# Build the dictionary
count = [("UNK", -1)] # Replace rare words with UNK token i.e. unknown
count.extend(collections.Counter(text_words).most_common(maximum_vocabulary_size - 1)) # Retrieve the most common words

for each_index in range(len(count) - 1, -1, -1):
    if count[each_index][1] < minimum_occurrence:
        count.pop(each_index)
    
    else:
        break # The collection is ordered, so stop when 'minimum_occurrence' is reached

vocabulary_size = len(count) # Compute the vocabulary size
word_to_id = dict() # Assign an id to each word

for each_index, (each_word, _) in enumerate(count):
    word_to_id[each_word] = each_index

data = list()
unknown_count = 0

for each_word in text_words:
    index = word_to_id.get(each_word, 0) # Retrieve a word id, or assign it index 0 ('UNK) if not in dictionary

    if index == 0:
        unknown_count += 1

    data.append(index)

count[0] = ("UNK", unknown_count)
id_to_word = dict(zip(word_to_id.values(), word_to_id.keys()))

print(f"word count: {len(text_words)}")
print(f"unique words: {len(set(text_words))}")
print(f"vocabulary size: {vocabulary_size}")
print(f"10 most common words: {count[:10]}")

data_index = 0

# Generate training batch for the skip-gram model
def next_batch(batch_size, number_of_skips, skip_window):
    global data_index
    assert batch_size % number_of_skips == 0
    assert number_of_skips <= 2 * skip_window

    batch = np.ndarray(shape = (batch_size), dtype = np.int32)
    labels = np.ndarray(shape = (batch_size, 1), dtype = np.int32)
    span = 2 * skip_window + 1 # Get window size i.e. words left and right + current position
    buffer = collections.deque(maxlen = span)

    if data_index + span > len(data):
        data_index = 0

    buffer.extend(data[data_index:data_index + span])
    data_index += span

    for each_index in range(batch_size // number_of_skips):
        context_words = [each_word for each_word in range(span) if each_word != skip_window]
        words_to_use = random.sample(context_words, number_of_skips)

        for each_other_index, each_context_word in enumerate(words_to_use):
            batch[each_index * number_of_skips + each_other_index] = buffer[skip_window]
            labels[each_index * number_of_skips + each_other_index, 0] = buffer[each_context_word]
        
        if data_index == len(data):
            buffer.extend(data[0:span])
            data_index = span
        
        else:
            buffer.append(data[data_index])
            data_index += 1

    data_index = (data_index + len(data) - span) % len(data) # Backtrack a little to avoid skipping words in the end of a batch
    return batch, labels

with tf.device(device):
    embedding = tf.Variable(tf.random.normal([vocabulary_size, embedding_size])) # Create the embedding variable (each row represent a word embedding vector)
    nce_weights = tf.Variable(tf.random.normal([vocabulary_size, embedding_size])) # Construct the weight variable for NCE loss
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

def get_embedding(input):
    with tf.device(device):
        input_embedded = tf.nn.embedding_lookup(embedding, input) # Lookup the corresponding embedding vectors for each sample in the input
        return input_embedded

# Compute the average NCE loss for the batch
def nce_loss(input_embedded, target):
    with tf.device(device):
        target = tf.cast(target, tf.int64)
        loss = tf.reduce_mean(tf.nn.nce_loss(weights = nce_weights, biases = nce_biases, labels = target, inputs = input_embedded, num_sampled = number_of_samples, num_classes = vocabulary_size))
        return loss

# Compute the cosine similarity between input data embedding and every embedding vectors
def evaluate(input_embedded):
    with tf.device(device):
        input_embedded = tf.cast(input_embedded, tf.float32)
        input_embedded_normalised = input_embedded / tf.sqrt(tf.reduce_sum(tf.square(input_embedded)))
        embedding_normalised = embedding / tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keepdims = True), tf.float32)
        cosine_similarity = tf.matmul(input_embedded_normalised, embedding_normalised, transpose_b = True)
        return consine_similarity

# Optimisation process
def run_optimisation(input, target):
    with tf.device(device):
        with tf.GradientTape() as g: # Wrap computation inside a GradientTape for automatic differentiation
            embedding = get_embedding(input)
            loss = nce_loss(embedding, target)

        gradients = g.gradient(loss, [embedding, nce_weights, nce_biases]) # Compute gradients
        optimser.apply_gradients(zip(gradients, [embedding, nce_weights, nce_biases]))

# Words for testing
test_input - np.array([word_to_id[each_word] for each_word in evaluate_words])

# Run training for the 
import numpy as np
from emo_utils import *
np.random.seed(0)
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation, Bidirectional
from keras.layers.embeddings import Embedding
np.random.seed(1)

X_train, Y_train = read_csv('data/train_emoji.csv')
X_test, Y_test = read_csv('data/tesss.csv')
maxLen = len(max(X_train, key=len).split())

Y_oh_train = convert_to_one_hot(Y_train, C = 5) # or use to_categorical()
Y_oh_test = convert_to_one_hot(Y_test, C = 5)

word_to_index, index_to_words, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')

def sentences_to_indices(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()

    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this.

    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """

    m = X.shape[0]                                   # number of training examples

    # Initialize X_indices as a numpy matrix of zeros and the correct shape (≈ 1 line)
    X_indices = np.zeros((m,max_len))

    for i in range(m):                               # loop over training examples

        # Convert the ith training sentence in lower case and split is into words.
        sentence_words = X[i].lower().split()

        # Initialize j to 0
        j = 0

        # Loop over the words of sentence_words
        for w in sentence_words:
            # Set the (i,j)th entry of X_indices to the index of the correct word.
            X_indices[i, j] = word_to_index[w]
            # Increment j to j + 1
            j = j+1

    return X_indices


def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.

    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """

    vocab_len = len(word_to_index) + 1                  # adding 1 to fit Keras embedding (requirement)
    emb_dim = word_to_vec_map["cucumber"].shape[0]      # define dimensionality of your GloVe word vectors (= 50)

    # Initialize the embedding matrix as a numpy array of zeros of shape (vocab_len, dimensions of word vectors = emb_dim)
    emb_matrix = np.zeros((vocab_len,emb_dim))

    # Set each row "index" of the embedding matrix to be the word vector representation of the "index"th word of the vocabulary
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    # Define Keras embedding layer with the correct output/input sizes
    embedding_layer = Embedding(vocab_len,emb_dim,trainable=False)

    # Build the embedding layer, it is required before setting the weights of the embedding layer.
    embedding_layer.build((None,))

    # Set the weights of the embedding layer to the embedding matrix.
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer

def Emojify_V2(input_shape, word_to_vec_map, word_to_index):
    """
    Function creating the Emojify-v2 model's graph.

    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    model -- a model instance in Keras
    """
    # Define X_input as the input of the graph, it should be of shape input_shape and dtype 'int32'
    X_input = Input(shape = input_shape, dtype = 'int32')

    # Create the embedding layer pretrained with GloVe Vectors and Propagate sentence_indices through the embedding layer,
    X = pretrained_embedding_layer(word_to_vec_map, word_to_index)(X_input)

    # Use bidirectional RNN with LSTM
    #X = Bidirectional(LSTM(units = 128, return_sequences = True))(X)

    # LSTM layer with return sequences
    X = LSTM(units = 128, return_sequences = True)(X)

    #  Add dropout
    X = Dropout(0.5)(X)

    # Use bidirectional RNN with LSTM
    # X = Bidirectional(LSTM(units = 128))(X)

    # LSTM layer without return sequences
    X = LSTM(units = 128)(X)

    X = Dropout(0.5)(X)

    # Softmax layer with 5 output units
    X = Dense(5)(X)
    X = Activation('softmax')(X)

    model = Model(inputs = X_input, outputs=X)

    return model

# build, compile and train model
model = Emojify_V2((maxLen,), word_to_vec_map, word_to_index)
model.summary()
model.compile(optimizer='Adam', loss= 'categorical_crossentropy', metrics=['accuracy'])

X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
Y_train_oh = convert_to_one_hot(Y_train, C=5)

model.fit(X_train_indices, Y_oh_train,batch_size= 32, epochs=50, shuffle= True)

# Test model
X_test_indices = sentences_to_indices(X_test, word_to_index, max_len = maxLen)
Y_test_oh = convert_to_one_hot(Y_test, C = 5)
loss, acc = model.evaluate(X_test_indices, Y_test_oh)
print()
print("Test accuracy = ", acc)

# check the mislabelled examples
C = 5
y_test_oh = np.eye(C)[Y_test.reshape(-1)]
X_test_indices = sentences_to_indices(X_test, word_to_index, maxLen)
pred = model.predict(X_test_indices)
for i in range(len(X_test)):
    x = X_test_indices
    num = np.argmax(pred[i])
    if(num != Y_test[i]):
        print('Expected emoji:'+ label_to_emoji(Y_test[i]) + ' prediction: '+ X_test[i] + label_to_emoji(num).strip())

# Your prediction.
x_test = np.array(['I have a headache'])
X_test_indices = sentences_to_indices(x_test, word_to_index, maxLen)

print()
print(x_test[0] +' '+  label_to_emoji(np.argmax(model.predict(X_test_indices))))

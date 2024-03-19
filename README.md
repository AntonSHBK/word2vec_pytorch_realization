# Overview and Implementation of Word2Vec using PyTorch

![Word2Vec](https://community.alteryx.com/t5/image/serverpage/image-id/45458iDEB69E518EBA3AD9?v=v2)
- [Code](word2vec.ipynb);
- [Russian decryption](Word2Vec_RU.md);
- [Habr's article (Ru)]().

Requirements (`python3.x`):
- `torch`;
- `numpy`;

`Word2Vec` is a popular word embedding model proposed by Google researchers in 2013 (Tomas Mikolov). It transforms words from a text corpus into vectors of numbers in such a way that words with similar semantic meanings have close vector representations in multidimensional space. This makes `Word2Vec` a powerful tool for natural language processing (`NLP`) tasks such as sentiment analysis, machine translation, automatic summarization, and many others.

## Key Features of Word2Vec:
* Distributed representation: Each word is represented as a vector in a multidimensional space, where relationships between words are reflected through the cosine similarity between their vectors.
* Unsupervised learning: Word2Vec is trained on large unlabeled text corpora without the need for external annotations or labeling.
* Contextual learning: Word vectors are obtained based on the context in which these words appear, capturing their semantic and syntactic relations.
## Two Main Model Architectures of Word2Vec:
`CBOW` (Continuous Bag of Words): This approach predicts the current word based on the context around it. For example, for the phrase "blue sky above the head", the `CBOW` model would try to predict the word "sky" based on the context words "blue", "above", "head". `CBOW` processes large volumes of data quickly but is less effective for rare words.

`Skip-Gram`: In this approach, the current word is used to predict the words in its context. For the same example, the `Skip-Gram` model would try to predict the words "blue", "above", "head" based on the word "sky". `Skip-Gram` processes data more slowly but works better with rare words and less common contexts.

## CBOW (Continuous Bag of Words)
The goal of `CBOW` is to predict the target word based on the context around this word. The context is defined as a set of words around the target word within a given window. The model architecture is simplified as a three-layer neural network: an input layer, a hidden layer, and an output layer.

![CBOW](https://www.researchgate.net/profile/Raouf-Ganda/publication/318975052/figure/fig2/AS:631670868820002@1527613479312/CBOW-architecture-predicts-the-current-word-based-on-the-context.png)

__Input layer:__ The model receives context words. These words are represented as vectors using "one-hot encoding", where each vector has a dimension equal to the size of the vocabulary and contains 1 at the position corresponding to the word's index in the vocabulary, and 0 in all other positions.

__Hidden layer:__ The input word vectors are multiplied by a weight matrix between the input and hidden layer, resulting in a hidden layer vector. For CBOW, the context word vectors are usually averaged before being passed to the next layer.

__Output layer:__ The hidden layer vector is multiplied by a weight matrix between the hidden and output layer, and the result passes through a `softmax` function to obtain probabilities of each word in the vocabulary being the target word. The goal of training is to maximize the probability of the correct target word.

### Skip-Gram
Unlike CBOW, the goal of Skip-Gram is to predict the context words for a given target word. This word at the model's input is used to predict words in its context within a given range of words (called the window).

![Skip-Gram](https://www.researchgate.net/profile/Firas-Odeh/publication/327537608/figure/fig5/AS:668724143063056@1536447668875/word2vec-Skip-gram-model-Image-credit-User-Moucrowap-on-Wikipedia.ppm)

__Input layer:__ The input is the target word, represented as a `one-hot` vector.
Hidden layer: The same as in CBOW, where the target word vector is multiplied by a weight matrix leading to the hidden layer.

__Output layer:__ Unlike `CBOW`, where the output layer is a single `softmax`, in `Skip-Gram` each word in the context uses a separate `softmax`, meaning the model tries to predict each context word separately. The goal of training is to maximize the probability of real context words appearing for a given target word.
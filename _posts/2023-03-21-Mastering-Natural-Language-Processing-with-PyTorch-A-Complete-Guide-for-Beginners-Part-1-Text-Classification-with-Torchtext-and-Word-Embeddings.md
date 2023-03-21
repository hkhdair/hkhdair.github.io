<!-- #region id="9aga3Dt1Fln8" -->
#### Mastering Natural Language Processing with PyTorch: A Complete Guide for Beginners _ Part 1 - Text Classification with Torchtext and Word Embeddings
<!-- #endregion -->

<!-- #region id="QjYSxNAiI3tP" -->
Natural Language Processing (NLP) is a rapidly growing field with numerous applications in text classification, sentiment analysis, language translation, and more. PyTorch, one of the most popular deep learning frameworks, has been increasingly used in the development of NLP models. With PyTorch, researchers and developers can easily build and train deep learning models for processing natural language data. In this tutorial series, we will explore various neural network architectures for NLP tasks and demonstrate how PyTorch can be used to implement them.

One of the essential components in NLP models is the handling of text data. The torchtext library provides a simple and efficient way to load and preprocess text data for NLP tasks. We will be using torchtext library in today's tutorial to load and preprocess text data. Additionally, word embeddings are a fundamental part of NLP, and we will be using the embedding layer from PyTorch to encode the text data into dense vectors, which can be processed by the NLP model.

In this first part of the tutorial series, we will be focusing on the text classification task for IMDB movie reviews, text classification is a crucial part of many real-world applications, such as search engines, recommender systems, and chatbots. Therefore, understanding how to build accurate and efficient text classification models is essential in NLP. 

We will start by exploring word embeddings and their significance in NLP models. We will learn how to load and preprocess text data using torchtext, and then use the embedding layer from PyTorch to generate word embeddings. By the end of this tutorial, you will have a good understanding of word embeddings and be able to use them to encode text data for NLP tasks using PyTorch.
<!-- #endregion -->

<!-- #region id="nAk1ZAgcNHYo" -->
##### General steps for building NLP models
<!-- #endregion -->

<!-- #region id="kLweItTpJo84" -->
Generally, the main steps for building an NLP model can be summarized as follows:

1- Data preparation and Vocabulary construction: This involves acquiring and preprocessing raw text data to make it suitable for use in a machine learning model. It involves building a vocabulary of all the unique words (or tokens) in the training data, and mapping each word to a unique index. Additionally, creating the training, validation and test dataset.

2- Model design: This involves selecting an appropriate neural network architecture for the specific NLP task, such as an recurrent neural network (RNN), or transformer.

3- Training and Evaluation: This involves feeding the preprocessed text data into the neural network, and measuring the performance on test set and using the trained model for inference.

We will go through each of these steps in details in this tutorial.
<!-- #endregion -->

<!-- #region id="kj8xwIe7nesL" -->
#### Preparing Text Data and Vocabulary Construction
<!-- #endregion -->

<!-- #region id="rxmttI1FqDjv" -->
Preparing text data involves several steps to transform raw text into a format that can be processed by deep learning models. The general steps are as follows:

1- Standardize the text (standardization): It is important to standardize the text by converting it to lowercase, removing punctuations, and handling special characters. This makes it easier to process and ensures consistency in the data.

2- Tokenize the text (tokenization): Text is then split into smaller units called tokens. These tokens can be characters, words, or groups of words. Tokenization can be performed using different methods and libraries such as the get_tokenizer function from the torchtext library.

3- Index the tokens (indexing): Each token is assigned a unique index, and a mapping between the token and its index in the vocabulary of all tokens is created. This is done to convert each token into a numerical vector that can be processed by deep learning models.

4- Convert tokens to numerical vectors (encoding or embedding): Each token is converted to a numerical vector using its index in the mapping created in the previous step. The vectors can be created using different techniques such as one-hot encoding or word embeddings.

By following these steps, raw text can be transformed into a format that can be processed by deep learning models. We will see in this tutorial how to use `get_tokenizer()` function to tokenize a sentence and convert the tokens to numerical vectors using a vocabulary.
<!-- #endregion -->

<!-- #region id="VBfscthstf7d" -->
##### Preparing the IMDB movie reviews dataset
<!-- #endregion -->

<!-- #region id="UesrA2z6ucnd" -->
The initial step is to download the dataset from Andrew Maas' Stanford page and then extract the compressed file.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Za41WLJ-FVx2" outputId="7ba0aa48-d39b-4340-c6b0-0d6691418891"
!curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
!tar -xf aclImdb_v1.tar.gz
```

<!-- #region id="ZSEQHEnOvZh9" -->
After downloading the dataset, you will have a directory named aclImdb that contains both the train and test subdirectories. The train directory has two subdirectories named "pos" and "neg", each of which contains 12,500 text files. Each file contains the text body of a positive-sentiment or negative-sentiment movie review to be used as training data. Similarly, the test directory also has two subdirectories named "pos" and "neg", each of which contains 12,500 text files with positive and negative sentiment reviews respectively. Therefore, there are 25,000 text files for training and another 25,000 for testing. 

The dataset also contains a train/unsup subdirectory that we do not need for this task, so let's delete it.
<!-- #endregion -->

```python id="5DO1InIru5Ep"
!rm -r aclImdb/train/unsup
```

<!-- #region id="-fEXESpow9cc" -->
Let's have a look at the data we are working with, i.e. look at a sample raw text data (a movie review) from the training dataset:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="xF9FWWE7vvPg" outputId="887600c6-b15b-4e9d-bcba-0a5989d0d171"
!cat aclImdb/train/pos/10001_10.txt

# out:
# Brilliant over-acting by Lesley Ann Warren. Best dramatic hobo lady I have ever seen, and love scenes in clothes warehouse are second to none. The corn on face is a classic, as good as anything in Blazing Saddles. The take on lawyers is also superb. After being accused of being a turncoat, selling out his boss, and being dishonest the lawyer of Pepto Bolt shrugs indifferently "I'm a lawyer" he says. Three funny words. Jeffrey Tambor, a favorite from the later Larry Sanders show, is fantastic here too as a mad millionaire who wants to crush the ghetto. His character is more malevolent than usual. The hospital scene, and the scene where the homeless invade a demolition site, are all-time classics. Look for the legs scene and the two big diggers fighting (one bleeds). This movie gets better each time I see it (which is quite often).

```

<!-- #region id="l6vGJlNYyZyA" -->
Great! let's now read the text data from the training dataset directory. We will define a simple function to read the text data and return text and label pairs, and read the text data and labels from the train directory, as follows:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Jg4qRkAcxhUH" outputId="e5c5430b-885e-431a-cbba-006eecc33420"
import os
import pathlib

# Define a function to read the text data and return text and label pairs
def read_text_data(data_path):
    texts = []
    labels = []
    for label in ['pos', 'neg']:
        label_path = os.path.join(data_path, label)
        for text_file in os.listdir(label_path):
            with open(os.path.join(label_path, text_file), 'r', encoding='utf-8') as f:
                text = f.read()
            labels.append(1 if label == 'pos' else 0)
            texts.append(text)
    return texts, labels

# Path to the directory of the saved dataset
data_path = pathlib.Path("aclImdb")

# Read the text data and labels from the train directory
texts, labels = read_text_data(data_path/'train')

print(f'Successfully read {len(texts)} texts, and {len(labels)} labels from training dataset')

# out:
# Successfully read 25000 texts, and 25000 labels from training dataset
```

<!-- #region id="v5xAJBGzEM4C" -->
##### Processing the dataset with a text tokenizer and constrcut the vocabulary
<!-- #endregion -->

<!-- #region id="fZX15BXoTU50" -->
We will use the `get_tokenizer()` function in the torchtext library from `torchtext.data.utils`. This function can be used to create a tokenizer that will be used to preprocess the text data, it takes an argument that specifies the type of tokenizer to create, we will use in this case a `basic_english` tokenizer. This is a simple type of tokenizer that splits the text into words based on whitespace and punctuation marks, converts all words to lowercase (i.e. standardizing and tokenizing). 
<!-- #endregion -->

```python id="el7O0W8d-sG4"
from torchtext.data.utils import get_tokenizer

# Define a tokenizer function to preprocess the text
tokenizer = get_tokenizer('basic_english')
```

<!-- #region id="I0msMrr1zsvp" -->
Let's see a sample standardized and tokenized text:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="a4eTzHrIzr3M" outputId="93bc7fd2-ce71-4d10-ab1f-ae1c97764932"
tokenizer('HERE Is an Example ;')

# out:
# ['here', 'is', 'an', 'example']
```

<!-- #region id="Nea4QpiAXFYF" -->
Next, we'll define a way to numercalize the tokens that can be created from the previous tokenizer, in particular, we'll index the tokens and map them to the vocabulary constructed for the entire words in the text corpus (i.e. indexing).
<!-- #endregion -->

```python id="4Wb0ferwXYOL"
from torchtext.vocab import build_vocab_from_iterator

# Build the vocabulary from the text data
vocab = build_vocab_from_iterator(map(tokenizer, texts), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

# Define a function to numericalize the text
def numericalize_text(text):
    return [vocab[token] for token in tokenizer(text)]
```

<!-- #region id="UuCHfF_CaUId" -->
The `build_vocab_from_iterator()` allows us to build a vocabulary from an iterator, which can be useful when working with large text corpus.

`map(tokenizer, texts)` builds a vocabulary from an iterator that applies the tokenizer function to each text in the texts list that we created earlier. The `specials=['<unk>']` argument specifies that the special token `<unk>` should be included in the vocabulary, and `vocab.set_default_index(vocab['<unk>']` sets the default index of the vocabulary object vocab to the index of the `<unk>` token. This means that any token that is not present in the vocabulary will be replaced with the `<unk>` token during numericalization. 

We'll have some checks for the constructed vocabulary for our text set and word indexes below:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="nWnPzzbzxE0z" outputId="74ca511c-32a6-4dd0-d721-4b4599142baf"
# the length of the constructed vocab from the text set, 100683 unique tokens
print(len(vocab))

# checking the index of words that are present in the vocabulary
print(vocab(['here', 'is', 'an', 'example']))

# checking the index of a word that is not present in the vocabulary, returns 0, the index for <unk>
print(vocab['biblioklept'])

# out:
# 100683

# [131, 9, 40, 464]

# 0
```

<!-- #region id="Bs4MwfuI5qDV" -->
Now, let's prepare the training and validation dataset that we will be using in our model. Using PyTorch's Dataset and Dataloader functionality is an efficient way to manage your data and simplify your machine learning workflow. By creating a Dataset, you can easily store all of your data in a structured manner. Dataloader, on the other hand, allows you to iterate through the data, manage batches, and perform data transformations, among other tasks. Together, these tools can help make your data more manageable and streamline your machine learning pipeline.
<!-- #endregion -->

<!-- #region id="fI982XnNKB8B" -->
We will build a custom text dataset from the texts and labels we created earlier, to learn the basics of how to build a custom text dataset, follow the tutorial in [this link](https://towardsdatascience.com/how-to-use-datasets-and-dataloader-in-pytorch-for-custom-text-data-270eed7f7c00).
<!-- #endregion -->

<!-- #region id="2ieFrJsaIyTl" -->
We can numericalize the text in the train dataset in the same custom dataset classs by the help of `numericalize_text()` that we defined in the earlier step, we can modify the `__getitem__` method of the CustomTextDataset class to apply the `numericalize_text()` function to each text sample, as follows:
<!-- #endregion -->

```python id="KHPoB3Wz3pOL"
import torch
from torch.utils.data import Dataset, DataLoader, random_split

# Define a custom dataset class for the text data
class CustomTextDataset(Dataset):
    def __init__(self, texts, labels, vocab, numericalize_text):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.numericalize_text = numericalize_text

    def __getitem__(self, index):
        label = self.labels[index]
        text = self.texts[index]
        numericalized_text = self.numericalize_text(text)
        return numericalized_text, label

    def __len__(self):
        return len(self.labels)
```

<!-- #region id="byaEoKA_JXp1" -->
In the code above, we modified the `__getitem__` method to first extract the text and label from the dataset, then apply the numericalize_text function to the text sample to get a list of numericalized tokens. The numericalized tokens are then stored in the "Text" field of the sample dictionary along with the label, which is stored in the "Class" field. Finally, the sample dictionary is returned.

We also added the vocab and tokenizer arguments to the CustomTextDataset constructor so that the tokenizer and vocabulary can be passed to the dataset. This allows the numericalize_text function to use the vocabulary and tokenizer defined earlier in the code.
<!-- #endregion -->

<!-- #region id="DvyOjWSs_niA" -->
Let’s create the dataset now as an object of CustomeTextDataset class, and create validation set by setting apart 20% of the training dataset.
<!-- #endregion -->

```python id="5mw0xO9e9NR8"
# Create train and validation datasets
dataset = CustomTextDataset(texts, labels, vocab, numericalize_text)
train_size = int(len(dataset) * 0.8)
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
```

<!-- #region id="WCmtJMeSqjIc" -->
Now, we'll generate the data batch. To generate the data batch, `torch.utils.data.DataLoader` is used here. 

Before sending to the model, `collate_fn` function works on a batch of samples generated from `DataLoader`. The input to `collate_fn` is a batch of data with the batch size in `DataLoader`, and `collate_fn` processes them according to the data processing a custom pipeline that we'll define in the following function named `collate_batch`, as follows.
<!-- #endregion -->

```python id="MZMKCFwDIWK4"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch.nn.utils.rnn import pad_sequence

# preprocess the data with a collate function, and pads the input sequences to the maximum length in the batch:
def collate_batch(batch):
    label_list, text_list = [], []
    for (_text, _label) in batch:
        label_list.append(_label)
        processed_text = torch.tensor(_text)
        text_list.append(processed_text)
    padded_text = pad_sequence(text_list, batch_first=False, padding_value=1.0)
    return torch.tensor(label_list, dtype=torch.float64).to(device), padded_text.to(device)

# Create train and validation data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, collate_fn=collate_batch, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, collate_fn=collate_batch, batch_size=batch_size, shuffle=False)
```

<!-- #region id="GhFGTskjs21Q" -->
You can see the benefit o the custom defined `collate_batch()` function we defined above, we used it for instance to pad the sequences with `padding_value=1.0`, the benefit of padding the sequence with 1.0 in the previous step is to make all the input sequences have the same length, which is required by most deep learning models for NLP. Padding also helps to preserve the information at the beginning and end of the sequence, which can be important for some tasks. Padding with 1.0 is a common choice because it is a neutral value that does not interfere with other tokens. 

We need such a step because natural language sentences have variable lengths, but deep learning models expect fixed-size inputs. For example, if we want to use a recurrent neural network (RNN) to process a batch of sentences, we need to pad them to the maximum length in the batch so that they can be fed into the deep learning model as a matrix.

An example of padding is shown below:

Original sentences: `[“I really love this movie”, “It was terrible”, “The acting was good”]`

Tokenized sentences: `[[9, 14, 56, 11, 17], [10, 21, 88], [8, 45, 21, 32]]`

Padded sentences to the longest sentence in our list (with maxlen=5): `[[9, 14, 56, 11, 17], [10, 21, 88 ,1 ,1], [8 ,45 ,21 ,32 ,1]]`
<!-- #endregion -->

<!-- #region id="t6mzjlqgu8g6" -->
Let's make a last check to see how the data is stored in our `train_loader` batches:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="4XOQDXAisngA" outputId="ce52d070-4d41-4c22-af58-edc5dc1e5c0b"
label, text = next(iter(train_loader))
print(label.shape, text.shape)
print(label, text)

# out:
'''torch.Size([32]) torch.Size([922, 32])
tensor([0., 0., 1., 1., 0., 0., 1., 0., 0., 1., 0., 1., 0., 1., 1., 0., 0., 0.,
        0., 0., 0., 1., 1., 1., 1., 1., 0., 1., 0., 1., 1., 1.],
       device='cuda:0', dtype=torch.float64) tensor([[    5, 23466,   152,  ...,    44,     5,    10],
        [  183,     9,    11,  ...,   163,  8490,  2399],
        [    6,   939,  6015,  ...,  2104,  1207,    77],
        ...,
        [    1,     1,     1,  ...,     1,     1,     1],
        [    1,     1,     1,  ...,     1,     1,     1],
        [    1,     1,     1,  ...,     1,     1,     1]], device='cuda:0')'''
```

<!-- #region id="Q7g5b71ZwGKJ" -->
One last note before starting the modelling, we used `batch_first=False` in `pad_sequence` in our custom collate function above, and the benefit is to make the padded sequences have a shape of (sequence_length, batch_size), which is the expected input shape for the model in PyTorch.

Great, we are done with preprocessin our data, let's build our model now.
<!-- #endregion -->

<!-- #region id="GDrAw03IxO7O" -->
#### Model Design
<!-- #endregion -->

<!-- #region id="wkEEsz7jzIJG" -->
We'll create a simple model for text classification. The model is composed of an `nn.Embedding` plus a linear layer for the classication purpose.
<!-- #endregion -->

<!-- #region id="Al-KQSHw1zpg" -->
<strong>But, what is the embedding layer, why did we use it, and how does it work?</strong>

We could feed the list of tokenized sequences e,g. `[   8,  124, 3732,  ..., 1197,   71, 4635],[ 145, 2402,    1,  ...,   74,   57,    4],
        ...,` directly to our model classifier layer, but our model won't make sense of the meaning of the words in our sentences or the relationship between the words in the sequence. The technique that we will be using here to do so is called the word embeddings with the help of `Embedding` layer from `torch.nn`.

Word embedding is a technique where individual words are represented as real-valued vectors in a lower-dimensional space and captures inter-word semantics. Word embeddings can be used to store word meanings and retrieve them using indices, as well as to measure the similarity or distance between words based on their vectors. 

The benefit of word embeddings is that they can preserve syntactic and semantic information of words and reduce the dimensionality of text data compared to other methods such as one-hot encoding or bag-of-words.

To learn more about the advantages of word embeddings compared to other methods, see [this link](https://www.turing.com/kb/guide-on-word-embeddings-in-nlp).

One way to best learn the embeddings of the words is to use it along with our text classification task. The `nn.Embedding` layer makes this possible, and the backpropagation technique of model trainining will learn the best weights of the layer along with our task. 

The output of the nn.embedding layer in the example is a tensor of shape (sequence_length, batch_size, embedding_dim) that contains the embeddings for each token in the input text. It’s common to see
word embeddings that are 256-dimensional, 512-dimensional, or 1024-dimensional when dealing with very large vocabularies. In our case we'll set the output dimension as 100.

Hence, our model defintion will be as follows:


<!-- #endregion -->

```python id="s9PPZtYEvdCl"
from torch import nn
import torch.nn.functional as F

class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, output_dim)
        
    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.permute(1, 0, 2)
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1) 
        return self.fc(pooled)
```

<!-- #region id="xLU76hhd_w8U" -->
Our model is a simple text classification model that consists of two layers: an embedding layer and a linear layer. It first applies the embedding layer to get a low-dimensional representation of each token in the text. Then it permutes the dimensions of the embedded tensor to match the expected input shape of the average pooling layer. Next, it applies the average pooling layer to get a fixed-length representation of the whole text by averaging over all tokens. Finally, it passes this representation to the linear layer to get a single output value for each text. The model is an adaptation of the [FastText](https://arxiv.org/pdf/1607.01759.pdf#:~:text=This%20paper%20explores%20a%20simple%20and%20ef%EF%AC%81cientbaseline%20for,sentences%20among%20312Kclasses%20in%20less%20than%20a%20minute) model, which provided a simple and efficient way to perform text classification.

The structure and layers of the model are shown below:

Input text -> Embedding layer -> Average pooling -> Linear layer -> Output value
<!-- #endregion -->

```python id="aMd_8qEk8Q5D"
# Create an instance of the text classification model with the given vocabulary size, embedding dimension and output dimension

model = TextClassificationModel(vocab_size = len(vocab), embedding_dim = 100, output_dim = 1)
```

<!-- #region id="VcF7Xv81ESs_" -->
#### Train and Evaluate the Model
<!-- #endregion -->

```python id="Tw4iQSlQ9AyQ"
# Define a loss function based on binary cross entropy and sigmoid activation
criterion = nn.BCEWithLogitsLoss()
# Define an optimizer that updates the model parameters using Adam algorithm
optimizer = torch.optim.Adam(model.parameters())

# Move the model to the device (CPU or GPU) for computation
model = model.to(device)
```

<!-- #region id="i3LqgE1_F9Q_" -->
Train the model for 10 epochs, print the training and validation loss and accuracy for each epoch:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="ZUViMrHBF0wK" outputId="98147450-54f7-479b-d1be-dd7b796000b2"
for epoch in range(10):
  epoch_loss = 0
  epoch_acc = 0
  
  model.train()
  for label, text in train_loader:
      optimizer.zero_grad()
      predictions = model(text).squeeze(1)
      loss = criterion(predictions, label)
      
      rounded_preds = torch.round(
          torch.sigmoid(predictions))
      correct = (rounded_preds == label).float()
      acc = correct.sum() / len(correct)
      
      loss.backward()
      optimizer.step()
      epoch_loss += loss.item()
      epoch_acc += acc.item()

  print("Epoch %d Train: Loss: %.4f Acc: %.4f" % (epoch + 1, epoch_loss / len(train_loader), 
                                                  epoch_acc / len(train_loader)))

  epoch_loss = 0
  epoch_acc = 0
  model.eval()
  with torch.no_grad():
    for label, text in val_loader:
      predictions = model(text).squeeze(1)
      loss = criterion(predictions, label)
      
      rounded_preds = torch.round(torch.sigmoid(predictions))
      correct = (rounded_preds == label).float()
      acc = correct.sum() / len(correct)
      
      epoch_loss += loss.item()
      epoch_acc += acc.item()

  print("Epoch %d Valid: Loss: %.4f Acc: %.4f" % (epoch + 1, epoch_loss / len(val_loader), 
                                                  epoch_acc / len(val_loader)))
```

<!-- #region id="eEFbMSSCHaO9" -->
After 10 epochs of training, we get around 88.9% accuracy on the validation dataset (note, your result may vary). This is still a simple model, the code can be modified or extended by changing some parameters or adding more layers.

Let's see how it performs on the test dataset, but let's first prepare it as we did  for the training and validation datasets before.
<!-- #endregion -->

```python id="WVL0y4GwG1Cy"
# Read the text data and labels from the test directory
test_labels, test_texts = read_text_data(data_path/'test')

# Create a custom text dataset object for the test data using the vocabulary and numericalize function
test_dataset = CustomTextDataset(test_labels, test_texts, vocab, numericalize_text)

# Create a data loader for the test dataset
test_loader = DataLoader(test_dataset, collate_fn=collate_batch, batch_size=batch_size, shuffle=False)
```

```python colab={"base_uri": "https://localhost:8080/"} id="q1O2K_H-J5lQ" outputId="51fe0af3-70eb-4de8-ffd0-17f2521e8500"
test_loss = 0
test_acc = 0
model.eval()
with torch.no_grad():
  for label, text in test_loader:
    predictions = model(text).squeeze(1)
    loss = criterion(predictions, label)
    
    rounded_preds = torch.round(
        torch.sigmoid(predictions))
    correct = (rounded_preds == label).float()
    acc = correct.sum() / len(correct)

    test_loss += loss.item()
    test_acc += acc.item()

print("Test: Loss: %.4f Acc: %.4f" %
        (test_loss / len(test_loader), 
        test_acc / len(test_loader)))

# out:
# Test: Loss: 0.3270 Acc: 0.8808
```

<!-- #region id="ioI0sJaBKpMr" -->
We get an around 88.1% on the test dataset. Not bad!

Let's see how it works on a some randome input text data, we'll do some housekeeping jobs first:
<!-- #endregion -->

```python id="InCU1rILKbXK"
# Define a text pipeline function that tokenizes and numericalizes a given sentence using the vocabulary
text_pipeline = lambda x: vocab(tokenizer(x))

# Define a function that predicts the sentiment of a given sentence using the model
def predict_sentiment(model, sentence):
    model.eval()
    text = torch.tensor(text_pipeline(sentence)).unsqueeze(1).to(device)
    prediction = model(text)
    return torch.sigmoid(prediction).item()
```

```python colab={"base_uri": "https://localhost:8080/"} id="nN_2MgxoMDkC" outputId="7c63a47b-b60c-424a-b0d9-fdd3982c6c44"
sentiment = predict_sentiment(model, "Very bad movie")
sentiment

# out:
# 5.249739285455989e-26
```

```python colab={"base_uri": "https://localhost:8080/"} id="ulI-_-9yMMva" outputId="7a601284-3920-4d24-ad48-70238f11433c"
sentiment = predict_sentiment(model, "This movie is awesome")
sentiment

# out:
# 1.0
```

<!-- #region id="Afh927apMWZB" -->
Well done! Our model gives a very low score to a negative sentiment and gives 1 for a positive movie review.

Our model is doing great, let's save it for inference later:
<!-- #endregion -->

```python id="QvD1Wmu8MTH9"
torch.save(model.state_dict(), 'movieclassification-model.pt')
```

<!-- #region id="EI6dAsnsNJAA" -->
##### Conclusion
<!-- #endregion -->

<!-- #region id="7lPI_84ONMoU" -->
In this tutorial, we have learned how to use torchtext library to load and preprocess text data for text classification. We have also learned how to use `nn.Embedding` layer to generate word embeddings and encode text data into low-dimensional vectors. We have built a simple text classification model using PyTorch and trained it on the IMDB movie reviews dataset. We have seen how word embeddings can improve the performance of text analysis by capturing inter-word semantics.

In the next tutorial, we will explore another neural network architecture for text classification: the Transformer encoder. The Transformer encoder is a powerful model that uses attention mechanism to process sequential data. We will see how to build and train a Transformer encoder using PyTorch for the same task of text classification. Stay tuned!
<!-- #endregion -->

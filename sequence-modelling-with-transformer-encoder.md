# Sequence Modelling with Transformer Encoder

We shall begin our journey of understanding transformers by discussing the design, components and implementation of a transformer model that can be used for basic language modelling tasks such as text classification. At the end of this chapter, we will have trained a transformer model to take a sequence of text and perform sentiment classification (.ie positive or negative).&#x20;

This chapter assumes no prior experience in natural language processing. Therefore, we will dedicate some paragraphs to explaining basic tools needed for processing text data and how to set them up. In later chapters, we will discussed more advanced tasks and models such as text generation with autoregressive transformer models (i.e GPTs), while the architecture for that has some new components, the core transformer architecture for GPTs is largely the same as the transformer encoder we will be discussing here. So let's get to it!



### What is a Transformer Model

Just like any neural network, transformers take an input and product a desired output, but unlike other neural networks, they contain specialized attention modules that explicitly model the relationship between different parts of an input sequence. Additionally, they have a positional encoding mechanism that explicitly models the position of each element relative to one another.

To understand this intuitively, lets take a look at the following text examples.

> _**I need to go to the bank to deposit some money**_
>
> _**We set up our picnic on the bank of the river**_

The two text sequences both use the word `bank`, but due to the context, the meaning of `bank` is dictinctly different. In the first sentence, bank refers to a financial institution, in the second one, it refers to land alongside a river. The attention mechanism of transformers enable modelling this types of complex relationships by learning what each word means based on the context in which it is used.

To understand positional encoding, let's look at the following examples.

> _**The stranger barked at the dog running quickly.**_
>
> _**Quickly, the dog barked at the running stranger.**_

The two text sequences above, are made up of the same words but, they mean two different things.

In the first one, we are talking about a stranger barking at a dog, the second one is talking about a dog barking at a stranger. While the words are same, the position and context changes meaning entirely.

As evidenced from the above examples, the ability to model the meaning of words based on both context and position is core to building a useful model of human language. Prior to tranformers, recurrent neural networks existed for this type of problems, however, they only worked well for short sequences, for much longer sequences, they failed to capture more complex contexts. Transformers are very verstatile for their ability to capture complex contexts in both short and long sequences, which is the reason why models like GPTs are possible. Without transformers, GPTs are simply not going to work well.

Furthermore, the principles discussed here extend to every single domain of artificial intelligence beyond text, in computer vision and speech processing, transformers are the dominant architecture because they enable processing images with proper context understanding and for speech, its trivial to  use transformers since the same principles that apply to text also applies to speech. For example, the same sound can indicate excitement or dissapointment depending on the context.&#x20;

In our pursuit of building intelligent systems that can see, hear, read, draw, write and speak, transformers is the key that makes it possible for us to build models that can understand the complexity of the world.



### Setup

We need to install a couple of things before we start discussing code and equations for transformers.

#### Install Python3

Install a recent version of python using conda or download it from [https://python.org](https://python.org)

#### Install Pytorch

We will be using pytorch for all experiments, go to [https://pytorch.org ](https://pytorch.org)for instructions on installing pytorch for your OS.

#### Install a few Extra Dependencies

In addition to pytorch, we will be using huggingface datasets library for loading datasets, as well as a few other libaries

```bash
pip3 install datasets tiktoken matplotlib
```



### NLP Prerequisite - Text Tokenization

Before we dive into the details of using transformers to process text, we need to understand the nature of text data and how neural networks are able to process it. Text is strings, however, neural networks are composed of mathematical operations which can only operate with numbers, either discrete (0,1,6,7) or continous ( 0.3, 1.67, 2.98).&#x20;

Therefore, we need a processes to convert text  (words, letters, symbols) into numbers before we can process them with a neural network. This process of turning text into numbers is called tokenization. You might notice we installed a python package named `tiktoken`, we will be using this extensively throughout this book for converting text to numbers. We will also explain how tokenizers work.

Below is an example of tokenizing text with tiktoken.

```python
""" Converting Text to Tokens """
import tiktoken

# create an instance of the GPT2 tokenizer 
tokenizer = tiktoken.get_encoding("gpt2")

text = "London is a beautiful city"

# tokenize your text
tokenized_text = tokenizer.encode(text)

# print result to see the tokens
print(tokenized_text)
```

When you run the above, you get the outcome below.

```bash
[23421, 318, 257, 4950, 1748]
```

These are called the tokens, which is the numbers representing the text, notice how we have 5 words in the source text `London is a beautiful city`, and 5 numbers in the token representation, `[23421, 318, 257, 4950, 1748]`, this is because, the text is splitted by the tokenizer into words and then each word is mapped to its equivalent integer, (we will discuss how this works under the hood shortly).

The cool thing is, you can take these tokens/numbers and convert them back into into the original text, the mapping is both ways. Let's put these to test.



```python
""" Converting Tokens to Text """
import tiktoken

# create an instance of the GPT2 tokenizer 
tokenizer = tiktoken.get_encoding("gpt2")

#the tokens we want to convert back to text
tokens = [23421, 318, 257, 4950, 1748]

converted_text = tokenizer.decode(tokens)

print(converted_text)
```

When you run the above, you get

```bash
London is a beautiful city
```

As you can see, it gave us back the exact text that produced those tokens in the first place, providing a consistent way to map text to numbers/tokens and tokens back to text.

#### How Tokenizers Work

Tokenizers are basically a giant dictionary of words and symbols mapped to integers. Predefined tokenizers such as the gpt2 tokenizer we have used above contain thousands of words mapped to arbitary numbers. We can define such a vocabulary ourselves as seen below.

```python
vocabulary = {
    "hello": 1,
    "world": 2,
    "this": 3,
    "is": 4,
    "a": 5,
    "test": 6
}
```

As you can see above, the assignment of words to integers is arbitary, the most important thing is each mapping has to be unique, no two words should map to the same integer and no two integers should map to the same word.

As long as we have such a unique dictionary of sufficiently large size, we can tokenize any text by.

1. Splitting it into words , e.g `this test` -> `[this, test]`
2. Convert the words into tokens using the vocabulary `[this, test]` -> `[3, 6]`&#x20;

We can convert back to the text through the reverse process.

Note also, the text isn't always split into words, typically sub-word tokenization is used, for example, a single word such as `building` will be split into two sub words, `build, ing` which will be then mapped to 2 tokens rather than 1.

We will pause our discussion of text tokenization there for now and get to explaining the various components of the transformer model.



### Components of a Transformer Encoder

<figure><img src=".gitbook/assets/image (2) (1).png" alt=""><figcaption><p>Fig 2.1 Illustration of of transformer model for topic classification</p></figcaption></figure>



The image above is a simplified illustration of how a transformer model takes a sequence of text and predicts some output, in this case, topic classification. Basically, the flow in the diagram above can be broken down below.

1. Convert the text into tokens using the tokenizer
2. Convert the tokens into embeddings using the embedding layer
3. Convert the position of each token into a positional encoding using the positional encoding layer
4. Add the token embedding and the positional embedding vectors
5. Pass the resulting vector through a list of N Feedforward + Attention layers. The attention layer learns the context and meaning of the words.
6. Pass the final output to the classification layer which will predict the final outcome such as the category of the text.

In the rest of this chapter, we shall explain each of this components in detail and implement them in pytorch. Finally, we shall put together all the layers and use it to construct a full transformer encoder model and use it to train a text classification model. Let's get to it!\


### Embedding Layer

The first layer of a transformer model is the embedding layer, this layer takes the tokens and converts each of them into a learned vector representation.

For example, once we have tokenized our text, "The goal was great", it becomes an array of integers like `X = [464, 3061, 373, 1049]` , mathematically, each of this integers/tokens is represented as $$x_i \subset R^1$$, which is just a fancy way of writing, each token $$x_i$$ in the set $$X$$ of all the input tokens to our transformer model is a real number with a single dimension (i.e, a scalar). In case this is confusing,&#x20;

a single number like `34` is represented as $$R^1$$ or simply $$R$$ while a vector like `[2.4, 7.9]` is represented as $$R^2$$

The embedding layer will project each token $$R^1$$ to a learned vector $$R^N$$, where N is the embedding dimension of the embedding layer.

Let's take a step back and discuss why we are doing this and what we are really doing by converting our tokens from their simple single number integer form to some more complex vector.

Remmember, during tokenization, we simply had a large dictionary of words predefined and we use that to map words and subwords to integers/tokens and back to words/subwords. This mapping is arbitary in the sense that, we can define `love` to map to 34, `affection` to map to 100 and `dislike` to 56, however this numbers have no implicit or explicit meaning, we could swap the numbers and it won't make any difference, for example, while love and affection are synonymous and dislike is the opposite of them, this relationships are not expressed by the integer value at all. Basically, the integer values of the tokens are meaningless. The embedding layer learns this meaning, during training, it maps the meaningless integer values into meaningful vector representations. For example, after training, the embedding vector $$R^N$$ of `love` and `affection` will be very similar, such that if you take the cosine similarity between the two vectors, it will be quite high, in the same way, the embedding vector $$R^N$$ of `love` and `dislike` will be quite disimilar such that if you compute the cosine similarity between the two vectors, it will be quite low. The key is, while tokenization makes it possible to turn text into numbers, the individual meaning of those words is learnt by the text embedding layer which turns the integers into embedding vectors. Note that we said, `individual` meanings, that is because, the meaning of the words can be changed due to context of the surrounding words. In order to learn the contextual meaning, we need the entire transformer model, with the attention layer being resposible for deriving the contextual meaning of each word and the entire sequence.

The embedding layer learns a table of mappings between the integer values and the vector representation, below is an illustration.

<figure><img src=".gitbook/assets/image (2).png" alt=""><figcaption><p>Fig 2.2, Illustration of the embedding table learned by the embedding layer</p></figcaption></figure>

In the above table, we have tokens 0 to token 32 000, this basically means the total size of our tokenizers vocabulary is 32, 001 words/subwords mapped to numbers 0 to 32, 000.&#x20;

Above, for each of theses numbers we learn a 6 dimensional vector, meaning we the embedding layer will project the token from  integer $$R^1$$  to vector $$R^6$$.

For example, token `31999` above, which in the tiktoken tokenizer maps to the word `extraordinarily`, has a learned 6 dimensional vector representation of `[0.96, 3.5, 4.56, 1.6, 2.8, 3.3]`,  this six dimensional vector is what we will pass to the other layers of our transformer model. Note, the above embeddings are just random values I made up to illustrate this, the values learned by the embedding layer of your trained model will be different from this and will be similar for similar words and quite different for different words.

The concept of embedding tokens into vector representations where similar tokens will be also similar in their vector representation and different tokens will be distinctly different in their vector representations, predates transformers. Word embeddings was largely popularized by the work of [Tomas Mikolov](https://arxiv.org/search/cs?searchtype=author\&query=Mikolov,+T) et al, in their landmark paper,[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)

They trained a couple of models to learn semantic vector representations of words in a continuos vector space.&#x20;

Now, let us see how to actually implement the embedding layer of the transformer in pytorch.



```python
import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):

        super().__init__()

        # Embedding layer maps discrete tokens to continous vectors
        # of dimension `embedding_dim`
        self.embedding_layer = nn.Embedding(
                                    num_embeddings=vocab_size, 
                                    embedding_dim=embedding_dim
                                )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens (torch.Tensor): Input tensor of shape (batch_size, seq_len).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embedding_dim).
        """

        # Get the embedding vector for the tokens
        embedding_vector = self.embedding_layer(tokens)

        return embedding_vector
```



The TokenEmbedding module above defines a pytorch module that contains an `nn.Embedding` layer with a vocabulary size equal to the total number of unique tokens that our model will support. This will be set to the size of our tokenizers dictionary, for example, if our tokenizer has a vocabulary size of 32, 000, we will set the number of embeddings in the embedding layer to be 32 000. This corresponds to the number of rows in `Fig 2.2` . The embedding dim defines the size of each embedding vector, this is the number of columns in `Fig 2.2` .&#x20;

In the forward function, we will pass in a bach of tokens, this will be of shape `[batch_size, seq_len]`, where `seq_len` is the number of tokens in our sequence. For example, `"The goal was great"` converts to 4 tokens  `[464, 3061, 373, 1049]`using the gpt2 tokenizer, therefore, the seq\_len is 4. In this function, the batch of tokens `[batch_size, seq_len]` we pass in will be passed down to the embedding layer which will then return a batch of vectors of shape, `[batch_size, seq_len, embedding_dim]`. As you can see, this changes every single integer token into a vector of dimension `embedding_dim.`

Below, we shall see this layer in action.

```python
# create an instance of the embedding layer
embedding_layer = TokenEmbedding(vocab_size=32_000, embedding_dim=6)
  
# create a token tensor of shape [1, 5], batch size 1, 5 tokens  
tokens = torch.tensor(
    [
        [101, 2, 3, 4, 5]
    ]
    )

# print the shape of the tokens to verify
print(tokens.shape)
```

```bash
torch.Size([1, 5])
```

```python
# pass the tokens to the token embedding layer
embedding_vector = embedding_layer(tokens)

#print the shape of the output
print(embedding_vector.shape)
```

```bash
torch.Size([1, 5, 6])
```

```python
# print the values of the output
print(embedding_vector)
```

```bash
tensor([[[-0.6032,  0.3096, -0.7081,  0.3671,  2.0060, -0.3745],
         [-0.4043, -0.4408, -0.5171,  0.6112, -0.3060,  0.0929],
         [ 1.4664, -1.7769, -0.5547,  1.4262,  0.5070, -1.0793],
         [-0.8065, -2.8148, -1.3158,  2.5726, -0.6245,  2.1777],
         [-1.8424, -0.8758, -1.7499,  0.9136,  1.0817,  0.6446]]],
       grad_fn=<EmbeddingBackward0>)
```

As you can see above, the token embedding layer has converted our simple list of tokens into a list of vectors. For example, the first token `101` has been converted to  vector`[-0.6032,  0.3096, -0.7081,  0.3671,  2.0060, -0.3745]`


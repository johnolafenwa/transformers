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












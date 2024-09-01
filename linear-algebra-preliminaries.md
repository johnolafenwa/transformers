# Linear Algebra Preliminaries

Understanding how transformers work requires some basic knowledge of linear algebra. These includes vector arithmetic, matrix operations and basic probability. To ease your understanding of the subsequent chapters in this book, we shall devote this chapter to discussing these fundamental math concepts, their equations and how to implement them in code. These should hopefully enable you comprehend some of the more mathematically involved aspects of the transformer architecture, specifically the attention mechanism.

## Data Types

There are three data types we will be using in building a transformer model. They are

* Scalars
* Vectors
* Matrices

Collectively, any of these datatypes can be described as a tensor, which in the context of neural networks is basically any collection of numbers of arbitary dimensions.

### Scalars

A scalar is simply a single number like `5` . It can be an integer or a float, basically, it is a tensor of a single dimension. Mathematically, a scalar can be represented as $$R^1$$ or simply $$R$$.

We can easily create a scalar tensor in pytorch as shown below.

```python
import torch

weight = torch.tensor([2.5])

print(weight.item())
```

This should give the output `2.5`

Notice above, a scalar can also be described as an array with just a single item.

Scalar operations including addition, subtraction, multiplication and division follows the same set of rules as any real valued number.&#x20;

Scalars can be of various types, common types include the following:

* **float32 (also just called float):** These is the default float datatype, the number is stored with 32 bits. 32 bit float is called single precision numbers. By default, neural network weights are stored as 32 bit float values.
* **float64 ( also called double):** These is a double precision number, each value is stored with 64 bits.  float64 is not typically used in training and running neural networks, they are useful for scientitic computing applications where extremely high precision is required.&#x20;
* **float16 (also called half):** These are numbers stored with 16 bits. They are the favourite choice for training and running large language models, this is because they require half the memory of float32 while still offering enough precision to train very accurate models.
* **int32:** This is the default datatype for all integers. It is stored with 32 bits and can contain both positive and negative integers. The values of an int32 data can range from $$-2^{31}$$ to $$2^{31} - 1$$
* **uint32 (unsigned int32):** These is same as int32 with 1 difference, it can only store positive integers. Its values can range from $$0$$ to $$2^{32} - 1$$
* **int64:** It is stored with 64 bits and can contain both positive and negative integers. The values of an int64 data can range from $$-2^{63}$$ to $$2^{63} - 1$$
* **uint64 (unsigned int64):** These is same as int64 but it can only store positive integers. Its values can range from $$0$$ to $$2^{64} - 1$$
* **int16:** It is stored with 16 bits and can contain both positive and negative integers. The values of an int16 data can range from $$-2^{15}$$ to $$2^{15} - 1$$
* **uint16 (unsigned int16):** These is same as int16 but it can only store positive integers. Its values can range from $$0$$ to $$2^{16} - 1$$

When working with these datatypes, it is important to keep in mind the implications of choosing a particular data type. A general consideration is smaller precisions such as float 16, int16 require less memory and are faster to compute than the larger precisions such as float32 and float64. In terms of memory usage, you can compute the memory usage of any datatype as follows.

Given that `8 bits = 1 byte`, a 16 bit float value will require a total memory of 16 `// 8 = 2 bytes` , similarly, a 64 bit float value will required a memory of `64 // 8 = 8 bytes`

These is quite useful for computing the total memory usage of a model. For example, to calculate the memory required by an LLM with 3 billion parameters, we can do the following.

Total bits = `3 billion * 16 bit = 48 billion bits`

Given `8 bits = 1 byte`

Total bytes = `48 billion // 8 = 6 Billion bytes = 6 GB`

Therefore, a 3 billion parameter model in 16 bit precision will occupy 6 GB of memory.



In pytorch, when creating a tensor, you can specify the data type as below.

```python
import torch
weight = torch.tensor([2.5], dtype=torch.float16)
```

### Vectors

A vector is an array of numbers. The elements of a vector are homogenous, for example, a vector can contain a number of floats, a number of integers, bools, long or short. However, a single vector cannot contain multiple elements of different types. As matter of fact, a vector always has a datatype, for example, the datatype of a vector of floats is float. Mathematically, a vector is written as $$R^N$$ where N is the number of elements in the vector, for example, a vector with 4 elements will be written as $$R^4$$

Below we shall create a vector in pytorch and print its datatype.

```python
import torch

vector = torch.tensor([1.5, -0.5, 3.0])

print(vector.dtype)
print(vector.shape)
print(vector)
```

These will output

```bash
datatype:  torch.float32
shape:  torch.Size([3])
content:  tensor([ 1.5000, -0.5000,  3.0000])
```

### Vector Operations

Below we will explore how to perform various operations with vectors

#### Vector Addition

You can add two vectors of the same dimension. This is done via element wise addition as seen below.

```python
import torch

vector_1 = torch.tensor([1.5, -0.5, 3.0])
vector_2 = torch.tensor([1.0, 2.0, 3.0])

# vector_sum = [1.5 + 1.0, -0.5 + 2.0, 3.0 + 3.0]
vector_sum = vector_1 + vector_2
print("vector_sum: ", vector_sum)
```

This will output

```bash
vector_sum:  tensor([2.5000, 1.5000, 6.0000])
```

Mathematically, this can be expressed as ;

Given two vectors A and B, the sum C = A + B of both vectors is computed elementwise as $$C_{i} = A_i + B_i$$. For example, `[2.0,  3.5] + [0.3, 1.0] = [2.0 + 0.3, 3.5 + 1.0] = [2.3, 4.5]`

Graphically, these looks like this

&#x20;$$\mathbf{A} = \begin{bmatrix} a_1 \\ a_2 \\ \vdots \\ a_n \end{bmatrix} \quad \text{and} \quad \mathbf{B} = \begin{bmatrix} b_1 \\ b_2 \\ \vdots \\ b_n \end{bmatrix}$$



$$\mathbf{C} = \mathbf{A} + \mathbf{B} = \begin{bmatrix} a_1 + b_1 \\ a_2 + b_2 \\ \vdots \\ a_n + b_n \end{bmatrix}$$

Note, you can only add two vectors to each other , their dimensions must be the same. For example, you can't add two vectors where one has 4 elements and the other has 6 elements.

One exception to this is adding a scalar to a vector, you can add a scalar to a vector by simply adding the number to every single element in the vector. Example below.

```python
import torch
# vector scalar addition
scalar = torch.tensor([2.0])
vector = torch.tensor([1.5, -0.5, 3.0])

# vector_scalar_add = [1.5 + 2.0, -0.5 + 2.0, 3.0 + 2.0]
vector_scalar_add = vector + scalar
print("vector_scalar_add: ", vector_scalar_add)
```

This will output

```bash
vector_scalar_add:  tensor([3.5000, 1.5000, 5.0000])
```

As you can see from the comment in the code, we simply added the scalar to every single element in the vector.

Graphically, we can represent vector scalar addition as

&#x20;$$\mathbf{Vector} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix} \quad \text{and} \quad \mathbf{Scalar} = x$$

&#x20;$$\mathbf{Vector + Scalar} = \begin{bmatrix} v_1 + x \\ v_2 + x \\ \vdots \\ v_n + x \end{bmatrix}$$

#### Vector Multiplication


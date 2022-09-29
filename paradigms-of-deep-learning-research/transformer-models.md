# Transformer Models

Until 2017, the three paradigms as highlighted above were Feedforward Networks which were used for very simple modelling tasks and as components of both convolutional neural networks and recurrent neural networks. And as earlier mentioned, convolutional neural networks were used primarily for computer vision and recurrent neural networks primarily for text and speech sequence modelling. All deep learning methods were based on these three paradigms.

In 2017, Vaswani et al published the ground-breaking paper, “_Attention is all You Need”**\[i]**_ , that brought the 4th paradigm of deep learning architectures.

These new models were called transformers and have since evolved in different ways achieving state of the art results on nearly every\
&#x20;task across computer vision, natural language processing, speech processing, and reinforcement learning. Beyond achieving SOTA results across these tasks, they have opened new frontiers in our ability to scale deep learning models, build single models that can perform multiple tasks\[ii] and achieve results that were previously impossible. Therefore, we shall spend the rest of this book learning what transformers are, how they work, and how to train and deploy them.

At their core, transformers are made up of three major components, standard linear modules, a positional encoder module and the Attention modules that model the dependency between each position in a sequence and all other dependencies in the sequence in a parallel fashion without needing to maintain a hidden state\[JO1] . More importantly, for each input, it can learn how important each of the other inputs are to model it. For example, in order to predict the next word in the sequence “_I am coming home to watch the evening match”_ given the input “_I am coming home to watch the evening”,_ the attention module enables the module to learn that value of the word “_watch”_ should be given more importance in order to predict the target word _“match”_ more than the word _“the”._ This is an arbitrary example that illustrates just how the attention module learns which inputs are useful and related to performing a task.

In simple terms, the attention module helps the model learn to focus.

As humans, we do this all the time. For instance, when I go into the kitchen to prepare tea, I would likely pay more attention to the part of the kitchen where the kettle is located as compared to the part of the kitchen where the fridge is located. Such focus helps us to solve problems efficiently, enabling us to focus less on parts of our environments that are not immediately useful to what we are doing at a point in time, albeit without completely disregarding the other aspects of our environment, helping us to use information from them as necessary. This simple concept is what the authors of the transformer paper used to lay the foundation for the next evolution of deep learning research.

Unlike RNNs , transformer models do not have the forgetting problem.

The formulation of transformer models can be represented by the equation for autoregressive models

$$
y_i = f(x_i \ | \ x_0, \ ..... \ ,x_{i-1};\theta)
$$

Furthermore, in non-casual settings (e.g sequence classification), each data point can depend not only on the past but can also depend on future data points, giving rise to the following equation:

$$
y_i = f(x_i \ | \ x_0, \ ..... \ ,x_{n-1};\theta)
$$

If the above isn’t super clear, don’t worry, we shall explain how transformers works in clearer detail in the next chapter with emphasis on the structure of the architecture and the attention module.

\
\


***

\[i] Attention Is All You Need https://arxiv.org/abs/1706.03762

\[ii] Reed, Scott; Zolna, Konrad; Parisotto, Emilio; Sergio Gomez Colmenarejo; Novikov, Alexander; Barth-Maron, Gabriel; Gimenez, Mai; Sulsky, Yury; Kay, Jackie; Jost Tobias Springenberg; Eccles, Tom; Bruce, Jake; Razavi, Ali; Edwards, Ashley; Heess, Nicolas; Chen, Yutian; Hadsell, Raia; Vinyals, Oriol; Bordbar, Mahyar; de Freitas, Nando (12 May 2022). "A Generalist Agent". arXiv:2205.06175 \[cs.AI].

***

&#x20;\[JO1]Insert diagram to explain these

&#x20;\[JO2]Recurrent Neural Networks: Add full meaning

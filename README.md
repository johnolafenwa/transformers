# Introduction to Transformers



## Overview

It has been a transformational couple of years in the race to build Artificial Intelligence systems. Across the industry, we have witnessed an explosion of new applications of deep learning models driven by large language models and generative models. In the medical space, AlphaFold\[i] has opened new ways to accelerate the discovery of new drugs via its ability to predict new protein structures. GPT-3 \[ii] has been applied to a large variety of text generation tasks including summarization of documents, writing essays, chat bots, text completion and most famously, assisting programmers to write software as demonstrated by OpenAI Codex \[iii]. The list goes on and on with the most recent popular application being diffusion models such as Dalle 2 \[iv] and Stable Diffusion \[v] whose unparalleled ability to create stunning images and art works has created a whole new category of artists and designers whose speciality is writing creative prompts for AI models to do the art work. To add to the growing list of miracle models we are seeing, Open AI recently published whisper, a single model that can translate speech from any language and accent to English text.

The fascinating thing about the above mentioned models is they are all based on a new paradigm of deep learning models called transformers. I have followed closely the literature of transformers for a past couple of years and it has been fascinating to see how they have evolved. Hence, after a long time of procrastinating, I am compelled to write this book to explain how and why transformers work so well.

Many materials exist today on transformers, most of this resources focusses mostly on the attention mechanism, in this book, my aim is to provide a single resource that greatly simplifies how transformers work, making them easy to understand and use. To achieve this, I shall explain both the theory and also provide practical code examples demonstrating how to solve a number of tasks using transformer models.

Furthermore, this book aims to explain transformers beyond natural language processing by delving into their applications in other fields including computing vision, speech and reinforcement learning.

My hope is, by the end of this treatise, the reader would have solid understanding of transformer models and how to apply them to solve a number of problems.

\


***

\[i] Highly accurate protein structure prediction with AlphaFold [https://www.nature.com/articles/s41586-021-03819-2](https://www.nature.com/articles/s41586-021-03819-2)

&#x20;

\[ii] Language Models are Few-Shot Learners [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)

&#x20;

\[iii] Evaluating Large Language Models Trained on Code [https://arxiv.org/abs/2107.03374](https://arxiv.org/abs/2107.03374)

&#x20;

\[iv] Hierarchical Text-Conditional Image Generation with CLIP Latents [https://arxiv.org/abs/2204.06125](https://arxiv.org/abs/2204.06125)

&#x20;

\[v] High-Resolution Image Synthesis with Latent Diffusion Models [https://arxiv.org/abs/2112.10752v2](https://arxiv.org/abs/2112.10752v2)

&#x20;


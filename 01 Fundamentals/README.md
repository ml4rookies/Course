# Machine Learning Fundamentals

  * [Artificial Intelligence](#Artificial-Intelligence)
    * Definition: overall introduction using the fourth quadrants, and then general categories problem solving, knowledge, reasoning and planning and learning and perciving and acting i.e. robotics
    * Foundation
    * History
  * Machine Learning
    * Solving problem without algorithms
      * Supervised vs unsupervised
      * Parametric vs nonparametrics (i.e automl) learning
    * Fundamental topics
      * Neural Networks
      * Gradient Descent
      * Backpropagation
      * Feature Engineering
      * Overfitting and underfitting: from memorization to generalization
      * Activation Function
    * Machine Learning Workflow
    * Popular examples
      * Sequential 
      * LSTM
    * Federated Learning
  * Tensorflow 2.0
    * What is Tensorflow?
      * Easy model building
      * Robust ML production anywhere
      * Powerful for research
    * Keras
      * Sequential API
      * Functional API
    * Components
      * Tensorflow Datasets
      * Tensorflow Hub
      * Model Garden
    * Use cases

## Artificial Intelligence

Have you meet intelligent entities?
  * We have been looking for Artificial Intelligent entities since 1957.
  * Actually, since then AI has been always a hot topic for scientist and engineers

But what are they? Based on [S. Russell and P. Norvig 2016](../05%20References/README.md)
**Human Performance** | **Human Rationality**
----------------- | ------------------
Thinking Humanly  | Thinking Rationally
Acting Humanly    | Thinking Rationally

According to the [Wikipedia entry](https://en.wikipedia.org/wiki/Artificial_intelligence), The field of AI research was born during a workshop at Dartmouth College in 1956.
The term "Artificial Intelligence" was coined by John McCarthy, inventor of Lisp programming language. 
The list of participants in that workshop includes several of the most important computer scienctist ever, 
such as Marvin Minsky, Claude Shannon, Julian Bigelow, W McCulloch, John Nash or Herbert A. Simon.

Since then, the AI discipline has evolved, including now several categories covering:

  * **Reasoning and problem solving**, including deterministics algorithms but also heuristics and probabilistics approaches.
  * **Knowledge representation and expert systems**, with special importance of the semantic modeling of knowledge.
  * **Planning** activities for intelligent agents, cooperation and competition among them and emergent behavior i.e evolutionary algorithms and swarm intelligence. 
  * **Learning**, in particular, machine learning, deep learning and statistical learning. In this course, we will focused in deep learning.
  * **Natural language processing**, i.e. the computatinal ability to understand human language.
  * **Perception**, in particular speech recognition or computational vision, using sensors perciving i.e. feeling the elements of the world.
  * **Motion and Manipulation**, in particular apply to robotics.

In the last 60 years, there have been recurrent hypes about the artificial intelligence and machine learning.
The current significan development of the field might be related to several aspect:

  * An eficient training algorithms for neural networks. See [Rumelhart, D.E., Hinton, G.E. and Williams, R.J. 1986](../05%20References/README.md).
  * Cheap hardware and cloud platforms.
  * Open source and collaborative research, such as [Arxiv](https://arxiv.org/) or [Kaggle](https://www.kaggle.com/)
  * Masive digitalization of our world contributing to the production of huge amount of training data.
 
Back to the [Index](../README.md)
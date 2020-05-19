Data is the better half of any machine learning model. A general principle that guides the development of the world-class machine learning models that we see today is - “garbage in garbage out”. It implies that if the data on which your machine learning model is getting trained is of bad quality, the model is also going to be of bad quality. 

Sometimes the irregularities in data can cause a machine learning model to pick up on the wrong input signals and eventually it may start giving really unexpected results. One such irregularity is bias. If your data is biased towards a factor or a set of factors, the model that would be trained on it will likely to be biased as well. Let’s take an example to understand this more thoroughly. 

Bing Image Search produces the following results on the keyword “Healthy Skin” - 

![](images/image_1.gif)

From the above examples, it is sort of clear that the machine learning model that is running behind the image search engine has clearly picked up the wrong signals from the training data to develop what does it mean for healthy skin. From the training data, the model has likely picked up signals like young women, fair complexion to learn what does it mean for healthy skin whereas we know that healthy skin has got nothing to do with those things. This presents an example of data bias. 

While developing machine learning models for production, as a responsible machine learning practitioners, we should always try to mitigate issues like this. This issue is really not a technical issue but is an ethical one which is way more important to tackle. 

The above example is attributed to [this chapter](https://nbviewer.jupyter.org/github/fastai/fastbook/blob/master/02_production.ipynb) of the book [Deep Learning for Coders with Fastai and PyTorch](https://www.amazon.com/Deep-Learning-Coders-fastai-PyTorch/dp/1492045527) by Jeremy Howard and Sylvain Gugger.   

If you are interested to know more about bias in machine learning in general, you may find [these materials](https://developers.google.com/machine-learning/crash-course/fairness/video-lecture) helpful. 

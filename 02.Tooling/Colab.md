## Introduction to Colab

Colaboratory or Colab for short, allows to run and program Python in your browser, with no installations. Main features include:
* No configuration or setup required
* Free GPU and TPU access.
* Easy sharing of documents (notebooks), like if there were Google Docs or Sheets docs.
* Easy Github access.
* TensorFlow 2 already preinstalled.
* Targeted at students and research scientists. Colab is not targeted at enterprise production environments.

If you know the [Jupyter project](https://jupyter.org/), Colab is a **Jupyter notebook stored in Google Drive**. Using a Google account, you can create a new notebook just by going to [colab.research.google.com](colab.research.google.com). If you **_do not_** want the Colab Notebooks to get automatically saved in your Google Drive account, you can use Colab Scratchpads: https://colab.research.google.com/notebooks/empty.ipynb. 

Colab connects your notebook to a Cloud-based runtime, meaning you can execute Python without any setup. It also allows you to [connect to a local runtime](https://medium.com/@jasonrichards911/getting-local-with-google-colab-a4d69f373364) but in this post, we will mainly be focusing on the former one i.e. the Cloud-based online runtime. 

A notebook document is composed of cells, of two types: code and text cells. **Code cells** contain python code. **Text cells** allow to add headings, paragraph, images, lists, even mathematical formulas using Markdown. All cells are executed using CMD+ENTER.

Notebooks used in Colab follow .ipynb standard format, and can be shared using Google Drive, github or can download it to be used in other compatible frameworks, like Jupyter notebooks, Jupyterlab. Moreover, you can find Notebooks repositories aroudn internet, like the [TensorFlow Hub project](https://tfhub.dev/), where there are many notebooks available and can be directly open in Colab.

## Colab for data science and deep learning

Colab comes with **TensorFlow 2 preinstalled**. But you can use any other dee learnihng framework with python. Note however Colab does not support R or scala. Additionally, Colab can use and import any popular library to visualize data. For example, you can import `numpy` to generate data and visualize it using `matplotlib`.

```python
import numpy as np
from matplotlib import pyplot as plt

ys = 200 + np.random.randn(100)
x = [x for x in range(len(ys))]

plt.plot(x, ys, '-')
plt.fill_between(x, ys, 195, where=(ys > 195), facecolor='g', alpha=0.6)

plt.title("Sample Visualization")
plt.show()
```


| ![Using numpy and matplotlib with Colab](./colab-matplotlib.png) | 
|:--:| 
| *Figure: Using numpy and matplotlib with Colab* |


## Using Colab with GitHub

Opening and saving `.ipynb` files stored on GitHub with Colab is easy: just add your GitHub path to colab.research.google.com/github/. For example, [colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb](colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb) will directly load this `.ipynb` stored on GitHub. For convenience, you can also use the [Open in Colab](https://chrome.google.com/webstore/detail/open-in-colab/iogfkhleblhcpcekbiedikdehleodpjo?hl=en) Chrome extension

You can also easily save a copy of your Colab notebook to Github by using File > Save a copy to GitHub.

## Some things to consider while using Colab

- Don't use an accelerator like a GPU or a TPU until it is required. It prevents abuse of use. 
- After your work is done, always terminate the session that frees up the resources allocated. This will also help you to reduce the number of cool-downs. 

## Other notebook environemtns

* **CAIP notebooks** (or Google Cloud AI Platforms notebooks): is a Google Cloud platform product, a managed service for JupyterLab, enterprise-grade, built on top of DL VM images. It is executed in user projects, and is now part of AI Platform. 

* **Colaboratory Pro** (or Colab Pro): similar to Colab, is a Google Brain product, not part of Google Cloud. [Announced in February 2020](https://colab.research.google.com/notebooks/pro.ipynb), it's a paid-based product (USD 9.99 / month), which provides faster GPUs(T4, P100), high-memory VMs and longer runtimes (before runtimes were resetted every 12 hours). 

* **Kaggle Kernels**: Similar to what Colab already offers but [Kaggle Kernels](https://www.kaggle.com/kernels) make it easier to work with Kaggle datasets. 

Back to the [Index](../README.md)

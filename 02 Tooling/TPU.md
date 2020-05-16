The Tensor Processing Unit (TPU) is a custom ASIC chip, designed by Google, **mainly designed for training** neural networks, although can also do inference.
It's currently in production in several Google consumer products, like email, YouTube and others. It's also available in Google Cloud Platform and some research products like Colab [colab.md].
It can be used not only by TensorFlow, but also by other deep learning frameworks like [PyTorch](https://cloud.google.com/tpu/docs/tutorials/pytorch-pod).

But what's the difference between a CPU, a GPU and a TPU ? You can access to this [demo site](https://storage.googleapis.com/nexttpu/index.html) or to [this link](https://cloud.google.com/blog/products/ai-machine-learning) to know more, but here is a quick summary:
* CPU is a general purpose processor, very flexible, that can be used for many tasks (word processing, bank transactions, ...) but lacks of parallelism and is slow for deep learning because needs to access the memory every time a single calculation is done by the Arithmetic Logic Units (ALU, the CPU component that holds and controls multipliers and adders).
* GPU is able to perform massive parallelism, such as matrix multiplications in a neural network, making suitable for deep learning. But still a GPU is a general purpose processor and has the same problem as before: a GPU needs a access shared memory to read and store intermediate calculations, which has also implications in energy consumption accessing memory.
* TPU is not a general purpose processor, but a matrix processor specialized for neural network workloads. It uses a systolic array architecture, based on thousands of multipliers and adders and connect them to each other directly to form a large physical matrix. The TPU loads first the paramters from memory, then the data, and finally executes the matrix multiplication, performing addings and data passing. During the whole process of massive calculations and data passing, **no memory access** is required at all. This is why the TPU can achieve a high computational throughput on neural network calculations with **much less power consumption**.

Note TPUs are not suited for all type of deep learning models. For example: if your model is dominated by element-wise multiplication (different from matrix multiplication, which is the recommended for TPUs); workloads that require double-precission arithmetic, or custom TensorFlow operations written in C++ (for this, better use CPUs). More information can be found [here](https://cloud.google.com/tpu/docs/tpus?hl=en#when_to_use_tpus).


## TPU types

There are different TPU devices depending on their capability:
* **Single device TPU types**, which are independent TPU devices without direct network connections to other TPU devices in a Google data center. For example: TPU v2-8 and TPU v3-8, each one containing 8 TPU cores.
* **TPU pods**, which are clusters of TPU devices that are connected to each other over dedicated high-speed networks, i.e., more efficient than connecting several independent TPU together in a single VM. For example: TPU v2-32 to 512 where number (32 to 512) refers to number of cores in the pod, or TPU v3-32 to 2048 (up to 2048 TPUs! in the pod).


| ![Cloud TPU Pod v3](https://cloud.google.com/images/products/tpu/google-cloud-ai.png) | 
|:--:| 
| *Figure: Cloud TPU Pod v3. Source: cloud.google.com/tpu* |

Here is a table describing TPU v2-8 and TPU v3-8:

**TPU v2-8** | **TPU v3-8**
----------------- | ------------------
8 GiB of HBM for each TPU core | 16 GiB of HBM for each TPU core
One MXU for each TPU core | Two MXU for each TPU core
Up to 512 total TPU cores and 4 TiB of total memory in a TPU Pod | Up to 2048 total TPU cores and 32 TiB of total memory in a TPU Pod
45 Tflops/chip | 105 Tflops/chip
Available for free in Colab [6] | Pay-per-use in Google Cloud Platform


For comparison purposes nVidia P100 has 18 Tflops, while NVidia V100 has 112 Tflops.

Note TPU v2-8 **can be used for free** in Colab, while the rest can be used from Google Cloud Platform services, like Compute Engine, Kubernetes Engine, Dataproc or CAIP notebooks. Note TPU pods are only available with a 1 year or 3 year commitment.


## Architecture 

The TPU architecture is based on a systolic array, that contains  256 × 256 = total 65,536 ALUs to perform matrix multiplication operations of inputs and weights in parallel:

| ![Systolic array architecture](https://storage.googleapis.com/gweb-cloudblog-publish/original_images/Systolic_Array_for_Neural_Network_2g8b7.GIF) | 
|:--:| 
| *Figure: Systolic array architecture. Source: Google blog* |


Each TPU chip has two cores. Each TPU core has a HBM memory module, and also a scalar, vector, and matrix (MXU) units. The MXU is probably the most important part, and is capable of performing 16K multiply-accumulate operations in each cycle. 

While the MXU inputs and outputs are 32-bit floating point values, the MXU performs multiplies at reduced [bfloat16 precision](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/bfloat16.h). Bfloat16 is a 16-bit floating point representation that provides better training and model accuracy than the IEEE half-precision representation. Bfloat16 provides the same range as float32 (i.e. initial and end number is the same) but reduces the precission (some gaps between initial and end number). The advantage is that  machine learning apps care less about the precission, while using bfloat16 reduces complexitiy and memory needed by half.

## Edge TPU

The Coral Edge TPU is an inference accelerator, targetting at making inference at devices and IoT applications. It's optimized for vision applications and convolutional neural networks. It follows the same architecture as a Cloud TPU, but requires quantized TensorFlow Lite models.
More information can be found at [Coral web page](https://coral.ai/products/)

Back to the [Index](../README.md)


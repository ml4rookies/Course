The Tensor Processing Unit (TPU) is a custom ASIC chip, designed by Google, **mainly designed for training**, but also can do inference.
It's currently in production in several Google consumer products, like email, YouTube and others. It's also available in Google Cloud Platform and some research products like Colab [REFERENCE]

## TPU types

There are different TPU devices depending on their capability:
* Single device TPU types, which are independent TPU devices without direct network connections to other TPU devices in a Google data center. For example: TPU v2-8 and TPU v3-8, each one containing 8 TPU cores.
* TPU pods, which are clusters of TPU devices that are connected to each other over dedicated high-speed networks, i.e., not like if you connect several TPU together in a single VM. For example: TPU v2-32 to 512, or TPU v3-32 to 2048 where number (32, 512, 2048) refers to number of cores in the pod.

NOTE on pricing: only available with a 1 year or 3 year commitment.

TPU v2-8
* 8 GiB of HBM for each TPU core
* One MXU for each TPU core
* Up to 512 total TPU cores and 4 TiB of total memory in a TPU Pod
* 45 Tflops/chip
* Avaialble for free in Colab [REFERENCE]

TPU v3-8
* 16 GiB of HBM for each TPU core
* Two MXUs for each TPU core
* Up to 2048 total TPU cores and 32 TiB of total memory in a TPU Pod
* 105 Tflops/chip

For comparison purposes nVidia P100 has 18 Tflops, while NVidia V100 has 112 Tflops.

Ecah TPU chip has two cores. Each TPU core has a HBM memory module, and also a scalar, vector, and matrix (MXU) units. The MXU is probably the most important part, and is capable of performing 16K multiply-accumulate operations in each cycle. 

While the MXU inputs and outputs are 32-bit floating point values, the MXU performs multiplies at reduced [bfloat16 precision](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/bfloat16.h). Bfloat16 is a 16-bit floating point representation that provides better training and model accuracy than the IEEE half-precision representation. Bfloat16 provides the same range as float32 (i.e. initial and end number is the same) but reduces the precission (some gaps between initial and end number). The advantage is that  machine learning apps care less about the precission, while using bfloat16 reduces complexitiy and memory needed by half.

## TPU scalalability 

Communication inside the TPU si extremely fast, but outside to the CPU host is much slower.


## Training on TPU Pods

All TPU types use the same data-parallel architecture. The only change is that the parallelism increases from 8 cores to 2048 cores.

To take full advantage of larger numbers of TPUs, you must tune several training task parameters. Refer to [this document](https://cloud.google.com/tpu/docs/training-on-tpu-pods?hl=en#overview) to see the changes.

Note evaluation is neither supported nor cost-effective on TPU pods.

Back to the [Index](../README.md)


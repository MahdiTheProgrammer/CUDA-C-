# A Deep learning framework from scratch with CUDA/C++

## Features
All of the following features support batch input which alongside with gpu acceleration, provides a descent performance.
  ### Layers:


#### Conv2d

$$
\text{Output}(i,j,k) = \sum_m \sum_p \sum_q \text{Input}(i+p, j+q, m) \times \text{Kernel}(p, q, m, k)
$$




#### Linear

$$
\text{Output} = W \times \text{Input} + b
$$





#### ReLU

$$
\text{ReLU}(x) = \max(0, x)
$$


#### MaxPool2d

$$
\text{Output}(i,j,k) = \max_{(p,q) \in \text{window}} \text{Input}(i+p, j+q, k)
$$




#### Flatten

$$
\text{Flatten}(x) = \text{reshape}(x) \quad \text{(collapse all spatial dimensions into one)}
$$



#### Softmax

$$
\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$


---    
    
  ### Loss Functions (Working on it!):
  
#### MSE

$$
\text{MSE} = \frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2
$$

#### BCE

$$
\text{BCE} = -\frac{1}{N}\sum_{i=1}^N \left( y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right)
$$

#### Huber

$$
L_\delta(a) =
\begin{cases}
  \frac{1}{2}a^2 & \text{if } |a| \leq \delta \\
  \delta(|a| - \frac{1}{2}\delta) & \text{otherwise}
\end{cases}
$$

---

  ### Optimization Functions (Soon will be added):
  - SGD
  - Adam
  - RMSProp

  ### TODO
  - add RNN, LSTM, transformer support
  - Autograd

## Test Dataset
For testing and demonstrating the functionality of the library I will use the CIFAR-10 dataset (binary version). The dataset is available in:
```
cs.toronto.edu
```
<a href="https://www.cs.toronto.edu/~kriz/cifar.html" target="_blank">Click here for the dataset</a>
## Build Instructions
Note: In order to run this project an Nvidia GPU is required. Furthermore CudaToolkit and nvcc must be installed on the machine. <br>
You can build your own model in 
```
LIB/main.cpp
```
Before running you have to check your gpu architecture from the following list:

- Kepler (GTX 600, GTX 700 series)  
  `-arch=sm_30` or `-arch=sm_35`

- Maxwell (GTX 750, GTX 900 series)  
  `-arch=sm_50` or `-arch=sm_52`

- Pascal (GTX 10xx series, like 1080 Ti)  
  `-arch=sm_60` or `-arch=sm_61`

- Volta (Tesla V100)  
  `-arch=sm_70`

- Turing (RTX 20xx series, like 2080 Ti)  
  `-arch=sm_75`

- Ampere (RTX 30xx series, like 3080, 3090)  
  `-arch=sm_80`  
  (`-arch=sm_86` for newer models like RTX 3050)

- Ada Lovelace (RTX 40xx series, like 4080, 4090)  
  `-arch=sm_89` (for some 4090)  
  `-arch=sm_90` (for newest architectures)

- Hopper (H100)  
  `-arch=sm_90`
  
and replace your gpus -arch with the one in
```
LIB/Makefile
```

Finally you can run it using the following command
```
make
```



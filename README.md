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

## Usage


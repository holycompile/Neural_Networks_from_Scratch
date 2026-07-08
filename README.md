# Neural Network from Scratch (in Python)

Welcome to the **Neural Network from Scratch** repository! This project serves as a step-by-step educational guide to understanding the mathematical and structural foundations of deep learning. By building a neural network from first principles, this codebase transitions gradually from manual scalar arithmetic to matrix operations, batch processing, activation functions, and modular object-oriented abstractions using NumPy.

This repository documents the core implementation files situated directly in the root directory.

---

## 📂 Repository Structure

The core files in this repository are structured as follows:

```text
Neural_Network_from_Scratch/
│
├── Neuron1.py                         # Step 1: Manual 3-neuron layer calculation (directly indexed)
├── Neuron2.py                         # Step 2: Manual 3-neuron layer calculation (via unpacked weight lists)
├── Neuron3.py                         # Step 3: Single neuron using NumPy dot product
├── Layer_of_Neurons.py                # Step 4: Layer of 3 neurons (single sample) using np.dot
├── Transpose_usecase.py               # Step 5: Batch inputs matrix dot product requiring transposing (.T)
├── layering2.0.py                     # Step 6: Two sequential dense layers using manual np.dot & transpositions
├── dense_layer.py                     # Step 7: Modular object-oriented Layer_Dense class with nnfs spiral data
├── relu1.py                           # Step 8: Modular dense layer integrated with ReluActivation class
├── keepdims.py                        # Step 9: Tutorial on np.sum behavior (axes and keepdims=True)
├── softmax.py                         # Step 10: Standalone Softmax normalization implementation
├── 1st_Forward_Implement_No_Loss.py   # Step 11: Unified feedforward pipeline (Dense -> ReLU -> Dense -> Softmax)
└── prototype/                         # Historical prototype scripts directory
```

---

## 🧠 Step-by-Step Implementation & Architecture Diagrams

### Step 1: The Multi-Neuron Manual Layer
📄 **Files:** [Neuron1.py](file:///c:/Users/User/OneDrive/Desktop/Neural_Network_from_Scratch/Neuron1.py) & [Neuron2.py](file:///c:/Users/User/OneDrive/Desktop/Neural_Network_from_Scratch/Neuron2.py)

To calculate output for a fully connected layer containing 3 neurons, we multiply a single 4-dimensional input vector by the unique weights of each neuron and add their corresponding biases.

$$y_j = \sum_{i=1}^{n} (x_i \cdot w_{j,i}) + b_j$$

#### 📊 Architecture Diagram
```mermaid
graph LR
    Inputs["Inputs: 1.0, 2.0, 3.0, 2.5"] --> N1["Neuron 1 (Manual Sum)"]
    Inputs --> N2["Neuron 2 (Manual Sum)"]
    Inputs --> N3["Neuron 3 (Manual Sum)"]
    N1 --> Out1["Out 1: 4.8"]
    N2 --> Out2["Out 2: 1.21"]
    N3 --> Out3["Out 3: -0.58"]
    
    style Inputs fill:#2c3e50,stroke:#34495e,stroke-width:2px,color:#fff;
    style N1 fill:#0984e3,stroke:#74b9ff,stroke-width:2px,color:#fff;
    style N2 fill:#0984e3,stroke:#74b9ff,stroke-width:2px,color:#fff;
    style N3 fill:#0984e3,stroke:#74b9ff,stroke-width:2px,color:#fff;
    style Out1 fill:#27ae60,stroke:#2ecc71,stroke-width:2px,color:#fff;
    style Out2 fill:#27ae60,stroke:#2ecc71,stroke-width:2px,color:#fff;
    style Out3 fill:#27ae60,stroke:#2ecc71,stroke-width:2px,color:#fff;
```

---

### Step 2: Vector Math (NumPy Dot Product)
📄 **File:** [Neuron3.py](file:///c:/Users/User/OneDrive/Desktop/Neural_Network_from_Scratch/Neuron3.py)

Replaces manual loop iterations with the dot product of two 1D NumPy arrays, demonstrating a clean implementation of a single neuron.

#### 📊 Architecture Diagram
```mermaid
graph LR
    Inputs["1D Input Array (1x4)"] --> Dot["np.dot(input, weight)"]
    Weights["1D Weight Array (1x4)"] --> Dot
    Bias["Bias: 2.0"] --> Add["+"]
    Dot --> Add --> Out["Output: 4.8"]
    
    style Inputs fill:#2c3e50,stroke:#34495e,stroke-width:2px,color:#fff;
    style Weights fill:#2c3e50,stroke:#34495e,stroke-width:2px,color:#fff;
    style Dot fill:#0984e3,stroke:#74b9ff,stroke-width:2px,color:#fff;
    style Bias fill:#6c5ce7,stroke:#a29bfe,stroke-width:2px,color:#fff;
    style Add fill:#0984e3,stroke:#74b9ff,stroke-width:2px,color:#fff;
    style Out fill:#27ae60,stroke:#2ecc71,stroke-width:2px,color:#fff;
```

---

### Step 3: Layer of Neurons (Matrix Multiplication)
📄 **File:** [Layer_of_Neurons.py](file:///c:/Users/User/OneDrive/Desktop/Neural_Network_from_Scratch/Layer_of_Neurons.py)

Groups individual weight vectors of multiple neurons into a single 2D weight matrix ($W$) of shape `(3, 4)`. Computing the outputs is vectorized using matrix-vector multiplication with input vector ($x$).

$$\text{Output} = W \cdot x + b$$

#### 📊 Architecture Diagram
```mermaid
graph LR
    Inputs["Inputs x (4,)"] --> Dot["np.dot(W, x)"]
    Weights["Weights Matrix W (3x4)"] --> Dot
    Biases["Biases (3,)"] --> Add["+"]
    Dot --> Add --> Out["Outputs (3,)"]
    
    style Inputs fill:#2c3e50,stroke:#34495e,stroke-width:2px,color:#fff;
    style Weights fill:#2c3e50,stroke:#34495e,stroke-width:2px,color:#fff;
    style Dot fill:#0984e3,stroke:#74b9ff,stroke-width:2px,color:#fff;
    style Biases fill:#6c5ce7,stroke:#a29bfe,stroke-width:2px,color:#fff;
    style Add fill:#0984e3,stroke:#74b9ff,stroke-width:2px,color:#fff;
    style Out fill:#27ae60,stroke:#2ecc71,stroke-width:2px,color:#fff;
```

---

### Step 4: Batch Processing & Matrix Transposition
📄 **File:** [Transpose_usecase.py](file:///c:/Users/User/OneDrive/Desktop/Neural_Network_from_Scratch/Transpose_usecase.py)

To process a batch of inputs, the input vector becomes a 2D matrix ($X$) of shape `(n_samples, n_features)`. In order to align dimensions for dot-product multiplication with the weight matrix $W$ of shape `(n_neurons, n_features)`, we must transpose the weight matrix ($W^T$).

$$\text{Output} = X \cdot W^T + b$$

#### 📊 Architecture Diagram
```mermaid
graph LR
    Inputs["Inputs X (3x4 Batch)"] --> Dot["np.dot(X, W.T)"]
    Weights["Weights Matrix W (3x4)"] --> Dot
    Biases["Biases (3,)"] --> Add["+"]
    Dot --> Add --> Out["Outputs (3x3 Batch)"]
    
    style Inputs fill:#2c3e50,stroke:#34495e,stroke-width:2px,color:#fff;
    style Weights fill:#2c3e50,stroke:#34495e,stroke-width:2px,color:#fff;
    style Dot fill:#0984e3,stroke:#74b9ff,stroke-width:2px,color:#fff;
    style Biases fill:#6c5ce7,stroke:#a29bfe,stroke-width:2px,color:#fff;
    style Add fill:#0984e3,stroke:#74b9ff,stroke-width:2px,color:#fff;
    style Out fill:#27ae60,stroke:#2ecc71,stroke-width:2px,color:#fff;
```

---

### Step 5: Chaining Sequential Layers
📄 **File:** [layering2.0.py](file:///c:/Users/User/OneDrive/Desktop/Neural_Network_from_Scratch/layering2.0.py)

Passes batch inputs through two sequential layers. The outputs of the first layer serve as the input vector to the second layer.

#### 📊 Architecture Diagram
```mermaid
graph LR
    Inputs["Inputs X (3x4)"] --> L1["Layer 1: np.dot(X, W1.T) + B1"]
    L1 --> L2["Layer 2: np.dot(L1, W2.T) + B2"]
    L2 --> Out["Outputs (3x3)"]
    
    style Inputs fill:#2c3e50,stroke:#34495e,stroke-width:2px,color:#fff;
    style L1 fill:#0984e3,stroke:#74b9ff,stroke-width:2px,color:#fff;
    style L2 fill:#0984e3,stroke:#74b9ff,stroke-width:2px,color:#fff;
    style Out fill:#27ae60,stroke:#2ecc71,stroke-width:2px,color:#fff;
```

---

### Step 6: Object-Oriented Modularity
📄 **File:** [dense_layer.py](file:///c:/Users/User/OneDrive/Desktop/Neural_Network_from_Scratch/dense_layer.py)

Introduces the `Layer_Dense` class. The weights are initialized with shape `(n_inputs, n_neurons)` using `0.01 * np.random.randn(...)`, which avoids the need to transpose weights during forward passes:

$$\text{Output} = X \cdot W + b$$

It uses the `nnfs` library to generate a non-linear spiral dataset ($X, y$) to verify the class.

#### 📊 Architecture Diagram
```mermaid
graph LR
    Dataset["spiral_data (100 samples, 2 features)"] --> L1["Layer_Dense (2 inputs, 3 neurons)"]
    L1 --> Out["Outputs (300x3)"]
    
    style Dataset fill:#2c3e50,stroke:#34495e,stroke-width:2px,color:#fff;
    style L1 fill:#0984e3,stroke:#74b9ff,stroke-width:2px,color:#fff;
    style Out fill:#27ae60,stroke:#2ecc71,stroke-width:2px,color:#fff;
```

---

### Step 7: Rectified Linear Unit (ReLU) Activation
📄 **File:** [relu1.py](file:///c:/Users/User/OneDrive/Desktop/Neural_Network_from_Scratch/relu1.py)

Introduces non-linear mappings via a `ReluActivation` class. It enforces:

$$f(x) = \max(0, x)$$

This step feeds the linear output of `Layer_Dense` through `ReluActivation`.

#### 📊 Architecture Diagram
```mermaid
graph LR
    Linear["Linear Outputs: np.dot(X, W) + B"] --> ReLU["ReLU: np.maximum(0, input)"] --> Out["Activated Outputs"]
    
    style Linear fill:#0984e3,stroke:#74b9ff,stroke-width:2px,color:#fff;
    style ReLU fill:#e17055,stroke:#fab1a0,stroke-width:2px,color:#fff;
    style Out fill:#27ae60,stroke:#2ecc71,stroke-width:2px,color:#fff;
```

---

### Step 8: Understanding NumPy Summing Dimensions
📄 **File:** [keepdims.py](file:///c:/Users/User/OneDrive/Desktop/Neural_Network_from_Scratch/keepdims.py)

A diagnostic utility demonstrating how `np.sum(axis, keepdims=True)` behaves. This is a crucial concept for understanding how inputs are normalized during the softmax calculation.

#### 📊 Concept Diagram
```mermaid
graph TD
    A["Matrix A (3x3)"] --> S0["np.sum(axis=0) -> shape (3,)"]
    A --> S0K["np.sum(axis=0, keepdims=True) -> shape (1,3)"]
    A --> S1["np.sum(axis=1) -> shape (3,)"]
    A --> S1K["np.sum(axis=1, keepdims=True) -> shape (3,1)"]
    
    style A fill:#2c3e50,stroke:#34495e,stroke-width:2px,color:#fff;
    style S0 fill:#0984e3,stroke:#74b9ff,stroke-width:2px,color:#fff;
    style S0K fill:#6c5ce7,stroke:#a29bfe,stroke-width:2px,color:#fff;
    style S1 fill:#0984e3,stroke:#74b9ff,stroke-width:2px,color:#fff;
    style S1K fill:#6c5ce7,stroke:#a29bfe,stroke-width:2px,color:#fff;
```

---

### Step 9: Softmax Activation (Probability Distribution)
📄 **File:** [softmax.py](file:///c:/Users/User/OneDrive/Desktop/Neural_Network_from_Scratch/softmax.py)

Implements Softmax calculation for multi-class classification. Subtraction of the row maximum ($x_{\max}$) prevents exponent overflow:

$$S_{i,j} = \frac{e^{z_{i,j} - z_{i,\max}}}{\sum_{k} e^{z_{i,k} - z_{i,\max}}}$$

#### 📊 Architecture Diagram
```mermaid
graph LR
    In["Inputs (3x4)"] --> Exp["exp(input - max)"]
    Exp --> Sum["Sum along axis=1 (keepdims=True)"]
    Exp --> Div["Divide: Exp / Sum"]
    Sum --> Div
    Div --> Prob["Probabilities (3x4)"]
    
    style In fill:#2c3e50,stroke:#34495e,stroke-width:2px,color:#fff;
    style Exp fill:#0984e3,stroke:#74b9ff,stroke-width:2px,color:#fff;
    style Sum fill:#6c5ce7,stroke:#a29bfe,stroke-width:2px,color:#fff;
    style Div fill:#0984e3,stroke:#74b9ff,stroke-width:2px,color:#fff;
    style Prob fill:#27ae60,stroke:#2ecc71,stroke-width:2px,color:#fff;
```

---

### Step 10: Unified Feedforward Implementation
📄 **File:** [1st_Forward_Implement_No_Loss.py](file:///c:/Users/User/OneDrive/Desktop/Neural_Network_from_Scratch/1st_Forward_Implement_No_Loss.py)

Chains together the completed pieces into a single feedforward pipeline: 
Input Spiral Data $\rightarrow$ Dense Layer 1 $\rightarrow$ ReLU Activation $\rightarrow$ Dense Layer 2 $\rightarrow$ Softmax Output Activation.

#### 📊 Architecture Diagram
```mermaid
graph LR
    In["Inputs X (300x2)"] --> L1["Layer 1: Dense (2 -> 3)"]
    L1 --> Act1["ReLU Activation"]
    Act1 --> L2["Layer 2: Dense (3 -> 3)"]
    L2 --> Act2["Softmax Output Activation"]
    Act2 --> Prob["Probabilities (300x3)"]
    
    style In fill:#2c3e50,stroke:#34495e,stroke-width:2px,color:#fff;
    style L1 fill:#0984e3,stroke:#74b9ff,stroke-width:2px,color:#fff;
    style Act1 fill:#e17055,stroke:#fab1a0,stroke-width:2px,color:#fff;
    style L2 fill:#0984e3,stroke:#74b9ff,stroke-width:2px,color:#fff;
    style Act2 fill:#e17055,stroke:#fab1a0,stroke-width:2px,color:#fff;
    style Prob fill:#27ae60,stroke:#2ecc71,stroke-width:2px,color:#fff;
```

---

## 🚀 Getting Started

### Prerequisites

You need Python 3, NumPy, and the `nnfs` helper library installed:

```bash
pip install numpy nnfs
```

### Running the Code

To execute the completed feedforward neural network pipeline and view the output class probabilities:

```bash
python 1st_Forward_Implement_No_Loss.py
```

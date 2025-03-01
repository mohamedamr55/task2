# Understanding LSTM Step-by-Step

## Introduction
LSTMs (Long Short-Term Memory networks) are a type of recurrent neural network (RNN) designed to process sequential data. This project provides a basic numerical example of how an LSTM processes inputs without relying on external machine learning frameworks.

## What This Project Covers
- Explanation of LSTM cell computations
- Manual calculation of gates and cell state updates
- Generating a prediction based on processed sequential data

## Files Included
- `lstm_example_v4.py`: A Python script implementing LSTM calculations manually.
- `README_v5.md`: Documentation explaining the logic behind the implementation.

## How It Works
At each time step, the LSTM processes an input value and updates its cell state and hidden state based on pre-defined weights and biases. The main steps include:
1. **Forget Gate** - Determines which past information should be retained.
2. **Input Gate** - Decides what new information to add.
3. **Cell State Update** - Updates the memory of the cell.
4. **Output Gate** - Generates the hidden state for the next step.

## Example Calculation
Here’s a simple equation representing the forget gate in the LSTM:
```python
forget_gate = 1 / (1 + np.exp(-(Wf * input_val + Uf * hidden_state + bf)))
```
This equation applies the sigmoid activation function to compute how much of the previous cell state should be kept.

## Why This Matters
Understanding LSTMs at a mathematical level provides insights into how deep learning models process sequential data like text, time series, and speech. This example simplifies LSTM computations to highlight their core functionality without requiring TensorFlow or PyTorch.

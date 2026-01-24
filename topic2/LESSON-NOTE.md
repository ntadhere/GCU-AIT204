# Matrices, the human brain, and ANN

## ANN (Artificail Neural Network)
- Interconnected nodeds: Networks cconsist of artificial neurons (nodes)

### The Neuron
- Components of a Neuron:
    - Bias (b): Additional parameter that shifts the activation fuction
        - What is shift:
        - I move a chair = I shift the chair to a new position base on the x-axis
        - shifting position = adding a value to an existing position
        - When do we need to shift the position? the table have four legs and it is not balance, we add a piece of page -> mean that we shift the legs that change the position of the table compare to the ground -> so the table does not move anymore
        -> bias help the model get closer to the reality by trained with error
        -> we cannot change the structure of the glasses but we can change the distance of glasses with eyes or .... that is the bias
    - Inputs (x₁, x₂, ..., xₙ): Values from previous layer or input data
    - Weights (w₁, w₂, ..., wₙ): Parameters that determine the strength of each input
    - Summation Function: Computes weighted sum of inputs
    - Activation Function: Determines the output based on the weighted sum

## MNIST Handwritten Disigt Recognition

- We don't know what happen, so we try all the possibility until it come to an conclusion
"""Softmax."""
import numpy as np

scores = np.array([3.0, 1.0, 0.2])
# import matplotlib.pyplot as plt

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x),axis = 0)

print(softmax(scores).T)
print(softmax(scores/10))

# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])
print(scores)
plt.plot(x, softmax(scores).T, linewidth=2)

# Softmax properties 

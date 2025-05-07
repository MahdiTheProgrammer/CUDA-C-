import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Huber loss function
def huber_loss(y_true, y_pred, delta=1.0):
    e = y_true - y_pred
    is_small_error = np.abs(e) <= delta
    squared_loss = 0.5 * e**2
    linear_loss = delta * (np.abs(e) - 0.5 * delta)
    return np.where(is_small_error, squared_loss, linear_loss)

# Create a grid of true and predicted values
y_true = np.linspace(-5, 5, 100)
y_pred = np.linspace(-5, 5, 100)
Y_true, Y_pred = np.meshgrid(y_true, y_pred)

# Compute the Huber loss at each (y_true, y_pred) pair
Z = huber_loss(Y_true, Y_pred, delta=1.0)

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(Y_true, Y_pred, Z, cmap='viridis')

ax.set_xlabel('True Value')
ax.set_ylabel('Predicted Value')
ax.set_zlabel('Huber Loss')
ax.set_title('Huber Loss Surface')

plt.savefig('huber_3d_plot.png')
# plt.show()

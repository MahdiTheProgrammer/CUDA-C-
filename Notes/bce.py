import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Make grid
y_hat = np.linspace(0.001, 0.999, 500)  # predicted probability
y_true = np.linspace(0, 1, 500)            # true label: only 0 or 1 (binary)
print(len(y_true))
# Create meshgrid
Y_hat, Y_true = np.meshgrid(y_hat, y_true)

# Compute BCE loss
Z = - (Y_true * np.log(Y_hat) + (1 - Y_true) * np.log(1 - Y_hat))

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(Y_hat, Y_true, Z, cmap='viridis')

ax.set_xlabel('Predicted Probability (ŷ)')
ax.set_ylabel('True Label (y)')
ax.set_zlabel('BCE Loss')
ax.set_title('Binary Cross Entropy 3D Surface')

plt.savefig('bce_3d_plot.png')
# plt.show()

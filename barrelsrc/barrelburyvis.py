# Importing necessary libraries
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Function to plot the cylinder and the plane
def plot_cylinder_and_plane(r, h, a, b, c, d):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plotting the cylinder
    z = np.linspace(0, h, 100)
    theta = np.linspace(0, 2 * np.pi, 100)
    theta, z = np.meshgrid(theta, z)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    ax.plot_surface(x, y, z, color='cyan', alpha=0.5)

    # Creating a grid for the plane
    xx, yy = np.meshgrid(np.linspace(-r, r, 100), np.linspace(-r, r, 100))
    zz = (-d - a * xx - b * yy) / c

    # Plotting the plane
    ax.plot_surface(xx, yy, zz, color='orange', alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Setting the limits for better visualization
    ax.set_xlim([-r, r])
    ax.set_ylim([-r, r])
    ax.set_zlim([0, h])
    
    plt.show()

# Parameters of the cylinder and plane
r = 1  # radius
h = 2  # height
a, b, c, d = 1, 1, 1, -1  # plane equation ax + by + cz + d = 0

# Plotting the cylinder and plane
plot_cylinder_and_plane(r, h, a, b, c, d)

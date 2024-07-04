import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg


def random_cylinder(x1, x2, r, npoints):
    """
    Generates uniformly distributed points within the volume of a defined cylinder
    """
    x1 = np.array(x1)
    x2 = np.array(x2)
    axis = x2 - x1
    h = np.linalg.norm(axis)
    axis = axis / h
    axnull = scipy.linalg.null_space(np.array([axis]))
    axnull1 = axnull[:, 0]
    axnull2 = axnull[:, 1]
    # sqrt radius to get uniform distribution
    rand_r = r * np.sqrt(np.random.random(npoints))
    rand_theta = np.random.random(npoints) * 2 * np.pi
    rand_h = np.random.random(npoints) * h
    
    cosval = np.tile(axnull1, (npoints, 1)) * rand_r[..., None] * np.cos(rand_theta)[..., None]
    sinval = np.tile(axnull2, (npoints, 1)) * rand_r[..., None] * np.sin(rand_theta)[..., None]
    xyzs = cosval + sinval + np.tile(x1, (npoints, 1)) + rand_h[..., None] * np.tile(axis, (npoints, 1))
    return xyzs


def classify_points(points, a, b, c, d):
    """Function to classify points based on the plane"""
    above_plane = 0
    below_plane = 0
    for point in points:
        x, y, z = point
        value = a * x + b * y + c * z + d
        if value > 0:
            above_plane += 1
        else:
            below_plane += 1
    return above_plane, below_plane


def monte_carlo_volume_ratio(n_points, r, h, a, b, c, d):
    """Monte Carlo method to estimate volume ratio"""
    points = random_cylinder(n_points, r, h)
    above_plane, below_plane = classify_points(points, a, b, c, d)
    ratio = above_plane / below_plane
    return ratio

def plot_cylinder_and_plane(r, h, a, b, c, d):
    """Function to plot the cylinder and the plane"""
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


if __name__ == "__main__":
    # Parameters of the cylinder and plane
    r = 1  # radius
    h = 2  # height
    a, b, c, d = 1, 1, 1, -1  # plane equation ax + by + cz + d = 0

    # Number of random points to sample
    n_points = 1000000

    # Calculate the volume ratio
    volume_ratio = monte_carlo_volume_ratio(n_points, r, h, a, b, c, d)
    print("Estimated Volume Ratio:", volume_ratio)

    # Plotting the cylinder and plane
    plot_cylinder_and_plane(r, h, a, b, c, d)

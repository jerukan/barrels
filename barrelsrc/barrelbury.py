import numpy as np

# Function to generate random points within the cylinder
def generate_random_points(n_points, r, h):
    points = []
    for _ in range(n_points):
        z = np.random.uniform(0, h)
        theta = np.random.uniform(0, 2 * np.pi)
        rho = np.random.uniform(0, r)
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        points.append((x, y, z))
    return np.array(points)

# Function to classify points based on the plane
def classify_points(points, a, b, c, d):
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

# Monte Carlo method to estimate volume ratio
def monte_carlo_volume_ratio(n_points, r, h, a, b, c, d):
    points = generate_random_points(n_points, r, h)
    above_plane, below_plane = classify_points(points, a, b, c, d)
    ratio = above_plane / below_plane
    return ratio

# Parameters of the cylinder and plane
r = 1  # radius
h = 2  # height
a, b, c, d = 1, 1, 1, -1  # plane equation ax + by + cz + d = 0

# Number of random points to sample
n_points = 1000000

# Calculate the volume ratio
volume_ratio = monte_carlo_volume_ratio(n_points, r, h, a, b, c, d)
print("Estimated Volume Ratio:", volume_ratio)

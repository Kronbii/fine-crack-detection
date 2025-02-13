import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from src.MST import sort_corners


def interpolate_contour_points(x, y, y_values):
    """Interpolates contour points so that there is a point for every specified y-coordinate."""
    sorted_indices = np.argsort(y)
    x = x[sorted_indices]
    y = y[sorted_indices]

    interpolated_x = np.interp(y_values, y, x)
    interpolated_points = np.column_stack((interpolated_x, y_values))

    return interpolated_points


def interpolate_points(old_x, old_y):
    """Interpolate the corners using a B-spline interpolation."""
    tck, u = splprep(
        [old_x, old_y], s=0, k=3
    )  # Using cubic spline (k=3) for smoothness
    xi, yi = splev(np.linspace(0, 1, 100), tck)  # Generate 100 interpolated points
    return xi, yi


# Generate random points
num_points = 10
x = np.random.uniform(0, 10, num_points)
y = np.random.uniform(0, 10, num_points)
y_values = np.linspace(0, 10, 100)  # Fine y-values for interpolation

# Sort the points using MST-based ordering
points = np.column_stack((x, y))
sorted_points = sort_corners(points)
new_x = [x for x, y in sorted_points]
new_y = [y for x, y in sorted_points]

# Apply interpolation techniques
points1 = interpolate_contour_points(x, y, y_values)  # Linear Interpolation
points2_x, points2_y = interpolate_points(new_x, new_y)  # B-Spline Interpolation

# Plot the results
plt.figure(figsize=(8, 6))

# Scatter plot of original points
plt.scatter(x, y, color="black", marker="o", label="Original Points")

# Linear Interpolation (np.interp)
plt.plot(
    points1[:, 0],
    points1[:, 1],
    color="red",
    linestyle="--",
    label="Linear Interpolation",
)

# B-Spline Interpolation
plt.plot(
    points2_x, points2_y, color="blue", linestyle="-", label="B-Spline Interpolation"
)

# Labels and Legend
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Comparison of Linear vs. B-Spline Interpolation")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

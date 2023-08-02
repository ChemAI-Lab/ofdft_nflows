import distrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


key = jax.random.PRNGKey(1234)
sigma = jnp.array([0.2, 0.3])  
mu = jnp.array([0., 0.])  

# Create an array of x values and y values for plotting the 2D surface
x_vals = jnp.linspace(-5., 5., 100)
y_vals = jnp.linspace(-5., 5., 100)
x_grid, y_grid = jnp.meshgrid(x_vals, y_vals)
xy_points = jnp.stack([x_grid, y_grid], axis=-1)

dist_distrax = distrax.Laplace(mu, sigma)


pdf_vals = jnp.prod(dist_distrax.prob(xy_points), axis=-1)

print(pdf_vals)
print(pdf_vals.shape)


# Normalize the joint PDF to have the total area under the surface sum up to 1
pdf_vals /= jnp.trapz(jnp.trapz(pdf_vals, x=x_vals, axis=1), x=y_vals)




integral = jnp.trapz(jnp.trapz(pdf_vals, x=x_vals, axis=1), x=y_vals)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_grid, y_grid, pdf_vals, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Probability Density')
plt.show()


#print("Gradients with respect to mu and sigma (batched):")
#print(gradients_batched)
print("Integral of the joint PDF:", integral)

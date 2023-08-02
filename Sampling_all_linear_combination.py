import distrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from jax import random 


jax.config.update("jax_enable_x64", True)

batches = 100
#Make the 2D grid for plotting
x_vals = jnp.linspace(-5., 5.,100)
y_vals = jnp.linspace(-5., 5., 100) 
x_grid, y_grid = jnp.meshgrid(x_vals, y_vals)


seed = 42
key = jax.random.PRNGKey(seed)
mu = jnp.array([0., 0.])
sigma = jnp.array([0.1, 0.2])

dist_distrax = distrax.Laplace(mu, sigma)

samples = dist_distrax.sample(seed=key, sample_shape=(batches))

log_prob = dist_distrax.log_prob(samples)
exp_prob = jnp.exp(log_prob)
prob = dist_distrax.prob(samples) #same as exp_prob

# print(exp_prob.shape) 
# print(samples.shape)

#vmap_prob = jax.vmap(dist_distrax.prob)
# pdf_vals = vmap_prob(samples)

#pdf_vals /= jnp.trapz(jnp.trapz(pdf_vals, x=x_vals, axis=1), x=y_vals)
#print(vmap_prob)
#print(vmap_prob.shape)
assert 0 
# print(samples.shape)
# print(exp_prob.shape)
# print(log_prob.shape)
# print(exp_prob)
# print(exp_prob.shape) 
# prod = jnp.ones_like(exp_prob)


# for i in range(batches):
#     print(prod)
#     prod *= jnp.prod(exp_prob[i])

#print(prod)

# assert 0 

# pdf_vals /= jnp.trapz(jnp.trapz(pdf_vals, x=x_vals, axis=1), x=y_vals)
# integral = jnp.trapz(jnp.trapz(pdf_vals, x=x_vals, axis=1), x=y_vals)


# print("Integral under the surface:", integral)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_grid, y_grid, pdf_vals, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Probability Density')
plt.show()
# print("Integral under the surface:", integral)


assert 0 

# pdf_vals = jnp.ones_like(x_grid)

# # Create the Laplace distributions and compute the joint probability density
# for i in range(len(mu)):
#     dist_distrax = distrax.Laplace(mu[i], sigma[i])
#     pdf_vals *= jnp.expand_dims(dist_distrax.prob(xy_points[:, :, i]), axis=-1)

# # Sum over the last axis to combine the joint probabilities of the individual distributions
# sum = jnp.sum(pdf_vals, axis=-1)

# pdf_vals /= jnp.trapz(jnp.trapz(pdf_vals, x=x_vals, axis=1), x=y_vals)


# # Calculate the integral under the surface using numerical integration (numpy trapz)
# integral = jnp.trapz(jnp.trapz(pdf_vals, x=x_vals, axis=1), x=y_vals)
# print("Integral under the surface:", integral)
# assert 0 


# # Plot the 2D surface
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(x_grid, y_grid, pdf_vals, cmap='viridis')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('Probability Density')
# plt.show()




# mu = jnp.ones_like(x_grid)
# b = jnp.ones_like(x_grid)

#Samples
# key,subkey = jax.random.split(key)
# samples = jax.random.uniform(subkey, shape=(batches, 2))

# mu = mu.at[:].set(samples[:,0])
# b = b.at[:].set(samples[:,1])

# dist_Laplacian = distrax.Laplace(samples[:,0], samples[:,1])

# print(dist_Laplacian.sample(seed=key).shape)
# assert 0 

#print(dist_Laplacian.sample(seed=key).shape)




# for i in range(batches):
#  key, subkey = jax.random.split(key)
#  mu = mu.at[i].set(random.uniform(key))
#  b = b.at[i].set(random.uniform(subkey))
#  dist_Laplacian = distrax.Laplace(mu,b)
#  pdf_vals *= jax.vmap(dist_Laplacian.prob)(xy_points[:, :, i])
# print(pdf_vals.shape)



# Sum over the last axis to combine the joint probabilities of the individual distributions
#pdf_vals = jnp.sum(sum, axis=-1)


# Normalize the joint PDF to have the total area under the surface sum up to 1
#pdf_vals /= jnp.trapz(jnp.trapz(pdf_vals, x=x_vals, axis=1), x=y_vals)
#assert 0 
# Calculate the integral under the surface using numerical integration (numpy trapz)
#integral = jnp.trapz(jnp.trapz(pdf_vals, x=x_vals, axis=1), x=y_vals)

#print("Integral under the surface:", integral)

#Plotting
# plt.plot(x_vals, pdf_vals, label='Joint Probability Density')
# plt.show()
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(x_grid, y_grid, sum, cmap='viridis')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('Probability Density')
# plt.show()




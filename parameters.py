import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import lognorm, norm
from hyperparameters import SimArgs


# Weights
def normal_visualisation(data):
    # Fit normal distribution to the log of data
    mu, std = norm.fit(data)

    fig, ax = plt.subplots(1, 1, figsize=(18, 6))
    ax.hist(data, bins=50, density=True, alpha=0.6, color='b')

    # Plot fitted normal distribution
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    pdf_fitted = norm.pdf(x, mu, std)
    ax.plot(x, pdf_fitted, 'r', linewidth=2,
            label=f'Fitted normal\nmu={mu:.4f}, std={std:.4f}')

    ax.set_title('Fitting Normal Distribution')
    ax.set_xlabel('x')
    ax.set_ylabel('Normalized density')
    ax.legend()

def weight_generation(key: jnp.array, sim_params: SimArgs,
                      visualize_plot: bool = True) -> (jnp.array, jnp.array):
    key, subkey = jax.random.split(key)
    normal_dist = jax.random.normal(subkey,
                                    shape=(sim_params.n_out,
                                           sim_params.n_in
                                           )
                                    )
    w = normal_dist * sim_params.w_scale / jnp.sqrt(sim_params.n_in)
    if visualize_plot:
        normal_visualisation(w.flatten())
    if sim_params.pos_w:
        w = jnp.abs(w)
    return key, w

def weight_generation_1h(key: jnp.array, sim_params: SimArgs,
                      visualize_plot: bool = True) -> (jnp.array, jnp.array):
    key, subkey = jax.random.split(key)
    win = jax.random.normal(subkey, shape=(sim_params.n_h,
                                           sim_params.n_in)
                            )
    win = win * sim_params.w_scale / jnp.sqrt(sim_params.n_in)
    wrec = jax.random.normal(subkey, shape=(sim_params.n_h,
                                           sim_params.n_h)
                            )
    wrec = wrec * sim_params.w_scale / jnp.sqrt(sim_params.n_h)
    wout = jax.random.normal(subkey, shape=(sim_params.n_out,
                                           sim_params.n_h)
                            )
    wout = wout * sim_params.w_scale / jnp.sqrt(sim_params.n_h)

    if sim_params.pos_w:
        win = jnp.abs(win)
        wout = jnp.abs(wout)
    return key, [win, wrec, wout]

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

def weight_generation_r1(key: jnp.array, sim_params: SimArgs,
                      bias_enable: bool) -> (jnp.array, jnp.array):
    key, subkey_in, subkey_rec, subkey_out, subkey_bias = jax.random.split(key, 5)
    win = jax.random.normal(subkey_in, shape=(sim_params.n_h,
                                           sim_params.n_in)
                            )
    win = win * sim_params.w_scale / jnp.sqrt(sim_params.n_in)
    wrec = jax.random.normal(subkey_rec, shape=(sim_params.n_h,
                                           sim_params.n_h)
                            )
    wrec = wrec * sim_params.w_scale / jnp.sqrt(sim_params.n_h)
    wout = jax.random.normal(subkey_out, shape=(sim_params.n_out,
                                           sim_params.n_h)
                            )
    wout = wout * sim_params.w_scale / jnp.sqrt(sim_params.n_h)

    if sim_params.pos_w:
        win = jnp.abs(win)
    wout = jnp.abs(wout)
    if bias_enable:
        wb = jax.random.normal(subkey_bias, shape=(sim_params.n_h,))
        wb = wb * sim_params.w_scale / jnp.sqrt(sim_params.n_h)
        return key, [win, wrec, wout, wb]
    else:
        return key, [win, wrec, wout]


def tau_generation(key, tau_bar, layer_size, dt):
    key, subkey = jax.random.split(key)
    gamma_k = 3
    gamma_theta = tau_bar / 3
    tau = jax.random.gamma(subkey, a=gamma_k, shape=(layer_size,)) * gamma_theta
    alpha = jnp.exp(-dt/tau)
    return key, alpha


def init_MLSNN(key, layer_widths, sim_params):
    params = []
    for layer_id, (in_width, out_width) in enumerate(zip(layer_widths[:-1], layer_widths[1:])):
        key, subkey_in, subkey_rec, subkey_bias, subkey_tm, subkey_ts = jax.random.split(key, 6)
        win = jax.random.normal(subkey_in, shape=(out_width, in_width))
        win = win * sim_params.w_scale / jnp.sqrt(in_width)

        if layer_id == len(layer_widths)-2:
            if sim_params.pos_w:
                win = jnp.abs(win)
            params.append([win])
            continue
        wrec = jax.random.normal(subkey_rec, shape=(out_width, out_width))
        wrec = wrec * sim_params.w_scale / jnp.sqrt(out_width)
        if sim_params.bias_enable:
            wb = jax.random.normal(subkey_bias, shape=(out_width,))
            wb = wb * sim_params.w_scale / jnp.sqrt(out_width)
        _, alpha = tau_generation(subkey_tm, sim_params.tau_syn, out_width, sim_params.timestep)
        _, beta  = tau_generation(subkey_ts, sim_params.tau_mem, out_width, sim_params.timestep)
        if sim_params.pos_w:
            win = jnp.abs(win)
            if sim_params.bias_enable:
                wb = jnp.abs(wb)
        if sim_params.bias_enable:
            params.append([win, wrec, wb, alpha, beta])
        else:
            params.append([win, wrec, alpha, beta])
    return key, params
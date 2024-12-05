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

def weight_generation(key: jnp.array,
                      sim_params: SimArgs,
                      visualize_plot: bool = True) -> (jnp.array, jnp.array):
    n_in = sim_params.layer_widths[0]
    n_out = sim_params.layer_widths[1]
    key, subkey = jax.random.split(key)
    normal_dist = jax.random.normal(subkey, shape=(n_out, n_in))
    w = normal_dist * sim_params.w_scale / jnp.sqrt(n_in)
    if visualize_plot:
        normal_visualisation(w.flatten())
    if sim_params.pos_w:
        w = jnp.abs(w)
    return key, w

def weight_generation_r1(key: jnp.array, sim_params: SimArgs,
                        ) -> (jnp.array, jnp.array):
    assert len(sim_params.layer_widths) == 3, f'Only 3 layers are supported, got {len(sim_params.layer_widths)}'
    key, subkey_in, subkey_rec, subkey_out, subkey_bias = jax.random.split(key, 5)
    win = jax.random.normal(subkey_in, shape=(sim_params.layer_widths[1],
                                           sim_params.layer_widths[0])
                            )
    win = win * sim_params.w_scale / jnp.sqrt(sim_params.layer_widths[0])
    wrec = jax.random.normal(subkey_rec, shape=(sim_params.layer_widths[1],
                                           sim_params.layer_widths[1])
                            )
    wrec = wrec * sim_params.w_scale / jnp.sqrt(sim_params.layer_widths[1])
    wout = jax.random.normal(subkey_out, shape=(sim_params.layer_widths[2],
                                           sim_params.layer_widths[1])
                            )
    wout = wout * sim_params.w_scale / jnp.sqrt(sim_params.layer_widths[1])

    if sim_params.pos_w:
        win = jnp.abs(win)
    wout = jnp.abs(wout)
    if sim_params.bias_enable:
        wb = jax.random.normal(subkey_bias, shape=(sim_params.layer_widths[1],))
        wb = wb * sim_params.w_scale / jnp.sqrt(sim_params.layer_widths[1])
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

def xavier_uniform_init(key, dim_in, dim_out):
    print('Xavier init')
    key, subkey = jax.random.split(key)
    dim_sum = dim_in + dim_out if dim_out is not None else dim_in
    bound = jnp.sqrt(6) / jnp.sqrt(dim_sum)
    shape = (dim_out, dim_in) if dim_out is not None else (dim_in,)
    w = jax.random.uniform(subkey, shape=shape, minval=-bound, maxval=bound)
    return key, w

def kaiming_uniform_init(key, dim_in, dim_out):
    '''
    Use kaiming initialization for the weights with a gain of a=sqrt(5)
    For more details, see:
        - torch.nn.Linear source code: https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
        - torch issue explaining it: https://github.com/pytorch/pytorch/issues/57109
    :param key: jax.random.PRNGKey
    :param dim_in: int
    :param dim_out: int
    :return: jax.random.PRNGKey, jnp.array
    '''
    print('MinGRU init')
    key, subkey = jax.random.split(key)
    k = 1/jnp.sqrt(dim_in)
    bound = jnp.sqrt(k)
    shape = (dim_out, dim_in) if dim_out is not None else (dim_in,)
    w = jax.random.uniform(subkey, shape=shape, minval=-bound, maxval=bound)
    return key, w


def init_MLSNN(key, sim_params):
    params = []
    for layer_id, (in_width, out_width) in enumerate(zip(sim_params.layer_widths[:-1], sim_params.layer_widths[1:])):
        key, subkey_in, subkey_rec, subkey_bias, subkey_tm, subkey_ts = jax.random.split(key, 6)

        init_fn = {
            'xavier_uniform': xavier_uniform_init,
            'kaiming_uniform': kaiming_uniform_init
        }
        init_fn = init_fn[sim_params.init_type]

        _, win = init_fn(subkey_in, in_width, out_width)
        _, wrec = init_fn(subkey_rec, out_width, out_width)
        if sim_params.bias_enable:
            _, wb = init_fn(subkey_bias, out_width, None)

        if layer_id == len(sim_params.layer_widths)-2:
            if sim_params.pos_w:
                win = jnp.abs(win)
            params.append([win])
            continue

        _, alpha = tau_generation(subkey_tm, sim_params.tau_syn, out_width, sim_params.timestep)
        _, beta  = tau_generation(subkey_ts, sim_params.tau_mem, out_width, sim_params.timestep)
        alpha = jnp.clip(alpha, a_min=0.272531793, a_max=0.995)
        beta = jnp.clip(beta, a_min=0.272531793, a_max=0.995)
        if sim_params.pos_w:
            win = jnp.abs(win)
            if sim_params.bias_enable:
                wb = jnp.abs(wb)
        if sim_params.bias_enable:
            params.append([win, wrec, wb, alpha, beta])
        else:
            params.append([win, wrec, alpha, beta])
    return key, params
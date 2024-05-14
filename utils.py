import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers
import numpy as np


@jax.custom_jvp
def gr_than(x, thr):
    """ Thresholding function for spiking neurons. """
    return (x > thr).astype(jnp.float32)


@gr_than.defjvp
def gr_jvp(primals, tangents):
    """ Surrogate gradient function for thresholding. """
    x, thr = primals
    x_dot, y_dot = tangents
    primal_out = gr_than(x, thr)
    tangent_out = x_dot / (10 * jnp.absolute(x - thr) + 1)**2
    return primal_out, tangent_out

# placeholder for the LIF neuron model
def lif_recurrent(state, input_spikes):
    return 0

def prediction_rn(params, hp, in_spikes):
    net_dyn = []
    for layer_id, layer in enumerate(params):
        i = jnp.zeros((layer[0].shape[0],))
        v = jnp.zeros((layer[0].shape[0],))
        z = jnp.zeros((layer[0].shape[0],))
        net_dyn.append((i, v, z))

    state = ((params, net_dyn), hp)
    _, net_dyn_hist = jax.lax.scan(lif_recurrent, state, in_spikes)

    return net_dyn_hist[-1][1], net_dyn_hist[-1][2], net_dyn_hist # TODO: return the full history of the network dynamics is expensive
    # return  v: (bs, 150, 20), z: (bs, 150, 20), [(i, v, z) for all layers]


prediction_jv_rn = jax.jit(jax.vmap(prediction_rn, in_axes=(None, None, 0)), static_argnums=(1,))


def loss_fn(params, hp, in_spikes, gt_labels):

    v, _, _ = prediction_jv_rn(params, hp, in_spikes)
    out = jnp.max(v, axis=1)
    logit = jax.nn.softmax(out)
    loss = -jnp.mean(jnp.log(logit[jnp.arange(gt_labels.shape[0]), gt_labels]))

    pred = jnp.argmax(out, axis=1)
    acc = jnp.count_nonzero(pred == gt_labels) / gt_labels.shape[0]
    return loss, acc


def run_batch(params, hp, loader):
    batch0 = next(iter(loader))
    in_spikes, gt_labels = batch0
    (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, hp, in_spikes, gt_labels)
    return (loss, acc), grads
def run_inference(params, hp, batch0):
    in_spikes, gt_labels = batch0
    _, _, net_dyn_hist = prediction_jv_rn(params, hp, in_spikes)
    return net_dyn_hist


def clip_tau(opt):
    (get_params, opt_state, opt_update) = opt
    params = get_params(opt_state)
    # print(f'opt_state: {opt_state}')
    # print(f'opt_state: {len(opt_state)}')
    # print(f'opt_state[0]: {len(opt_state[0])}')
    # print(f'opt_state[0][0]: {len(opt_state[0][0])}')
    offset = len(params[0])
    for layer_id, layer in enumerate(params):
        if layer_id == len(params) - 1:
            break
        # print(f'clip: {layer_id}, {len(params)}, opt_state: {len(opt_state[0])}')
        layer[-2] = jnp.clip(layer[-2], a_min=0.272531793, a_max=0.995)  # exp(-1/3) = 0.272531793
        # print(f'f{((layer_id+1)*offset-2)=}: {jnp.min(layer[-2])=}')
        opt_state[0][(layer_id+1)*offset-2][0] = layer[-2]
        layer[-1] = jnp.clip(layer[-1], a_min=0.272531793, a_max=0.995)  # a_min=3*args.timestep, a_max=-3/jnp.log(0.995))
        # print(f'f{((layer_id+1)*offset-1)=}: {jnp.min(layer[-1])=}')
        opt_state[0][(layer_id+1)*offset-1][0] = layer[-1]
    return (get_params, opt_state, opt_update)


# def update(opt, hp, in_spikes, gt_labels, e, train_tau):
#     get_params, opt_state, opt_update = opt
#     w = get_params(opt_state)
#     (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(w, hp, in_spikes, gt_labels)
#     # check if the gradients are NaN
#     for id_g, grad in enumerate(grads):
#         if jnp.any(jnp.isnan(grad)):
#             print(f'{e}, {id_g}: NaN in gradients')
#             return np.nan
#     opt_state = opt_update(e, grads, opt_state)
#     opt = (get_params, opt_state, opt_update)
#     if train_tau:
#         opt = clip_tau(opt)
#     return (loss, acc), opt

def update(opt, hp, in_spikes, gt_labels, e, train_tau):
    get_params, opt_state, opt_update = opt
    params = get_params(opt_state)
    (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, hp, in_spikes, gt_labels)
    # check if the gradients are NaN
    # for layer_id, layer in enumerate(grads):
    #     for id_g, grad in enumerate(layer):
    #         if jnp.any(jnp.isnan(grad)):
    #             print(f'{e}, {layer_id}, {id_g}: NaN in gradients')
    #             return np.nan
    opt_state = opt_update(e, grads, opt_state)
    opt = (get_params, opt_state, opt_update)
    if train_tau:
        opt = clip_tau(opt)
    return (loss, acc), opt

def run_epoch(opt, hp, loader, e, train_tau):
    epoch_loss = jnp.zeros((len(loader)))
    epoch_acc = jnp.zeros((len(loader)))
    for id_batch, (in_spikes, gt_labels) in enumerate(loader):
        (loss, acc), opt = update(opt, hp, in_spikes, gt_labels, e, train_tau)
        epoch_loss = epoch_loss.at[id_batch].set(loss)
        epoch_acc = epoch_acc.at[id_batch].set(acc)
    return (epoch_loss, epoch_acc), opt


def inference(params, hp, loader):
    inference_loss = np.zeros((len(loader)))
    inference_acc = np.zeros((len(loader)))
    full_label = []
    full_pred = []
    for id_batch, (in_spikes, gt_labels) in enumerate(loader):
        v, _, _ = prediction_jv_rn(params, hp, in_spikes)
        out = jnp.max(v, axis=1)
        logit = jax.nn.softmax(out)
        inference_loss[id_batch] = -jnp.mean(jnp.log(logit[jnp.arange(gt_labels.shape[0]), gt_labels]))

        pred = jnp.argmax(out, axis=1)
        inference_acc[id_batch] = jnp.count_nonzero(pred == gt_labels) / gt_labels.shape[0]
        full_label.append(gt_labels)
        full_pred.append(pred)
    full_label = np.concatenate(full_label, axis=0)
    full_pred = np.concatenate(full_pred, axis=0)

    return inference_loss, inference_acc, (full_label, full_pred)


def train(w, hp, loaders, args):
    train_loader, val_loader, test_loader = loaders
    opt_init, opt_update, get_params = optimizers.adam(step_size=args.lr)
    opt_state = opt_init(w)
    print(f'{"Epoch":<6}|{"Loss":<10}|{"Acc":<10}|{"Val Acc":<10}|{"Test Acc":<10}|{"Val Loss":<10}|{"Test Loss":<10}')
    print(f'{"-" * 6}|{"-" * 10}|{"-" * 10}|{"-" * 10}|{"-" * 10}|{"-" * 10}|{"-" * 10}')
    best_val_acc = 0.01  # Random guess
    patience = 20
    best_opt_state = None
    hist_train_loss = np.zeros((args.nb_epochs))
    hist_val_loss = np.zeros((args.nb_epochs))
    hist_test_loss = np.zeros((args.nb_epochs))
    test_loss = 0
    opt = (get_params, opt_state, opt_update)
    for e in range(args.nb_epochs):
        (epoch_loss, epoch_acc), opt = run_epoch(opt, hp, train_loader, e, args.train_tau)
        _, opt_state, _ = opt
        if args.train_tau:
            params = get_params(opt_state)
            offset = len(params[0]) # number of parameters per layer (4 if no bias, 5 if bias)
            # print(f'offset: {offset}')
            for layer_id, layer in enumerate(params):
                if layer_id == len(params) - 1:
                    break
                # print(f'f{((layer_id+1)*offset-2)=}: {jnp.min(layer[-2])=}')
                # print(f'f{((layer_id+1)*offset-1)=}: {jnp.min(layer[-1])=}')

                if opt_state[0][(layer_id+1)*offset-2][0].min() < 0.272531793 or opt_state[0][(layer_id+1)-2][0].max() > 0.995:
                    print(f'{e}, -2, {len(opt_state[0][(layer_id+1)-2])}: {opt_state[0][(layer_id+1)-2][0]}')
                    break
                elif opt_state[0][(layer_id+1)*offset-1][0].min() < 0.272531793 or opt_state[0][(layer_id+1)*offset-1][0].max() > 0.995:
                    print(f'{e}, -1, {len(opt_state[0][(layer_id+1)*offset-1])}: {opt_state[0][(layer_id+1)*offset-1][0]}')
                    break
        val_loss, val_acc, _ = inference(get_params(opt_state), hp, val_loader)
        if val_acc.mean() > best_val_acc:
            best_val_acc = val_acc.mean()
            best_opt_state = opt_state
            patience = 20
            test_loss, test_acc, _ = inference(get_params(opt_state), hp, test_loader)
            print(
                f'{e:<6}|{epoch_loss.mean():<10.4f}|{epoch_acc.mean():<10.4f}|{val_acc.mean():<10.4f}|{test_acc.mean():<10.4f}|{val_loss.mean():<10.4f}|{test_loss.mean():<10.4f}')
        else:
            patience -= 1
            if e % 1 == 0:
                print(
                    f'{e:<6}|{epoch_loss.mean():<10.4f}|{epoch_acc.mean():<10.4f}|{val_acc.mean():<10.4f}|{"-":<10}|{val_loss.mean():<10.4f}|{"-":<10}')
            if patience == 0:
                break
        hist_train_loss[e] = epoch_loss.mean()
        hist_val_loss[e] = val_loss.mean()
        hist_test_loss[e] = test_loss.mean()

    test_loss, test_acc, _ = inference(get_params(best_opt_state), hp, test_loader)
    print(f'{e:<6}|{"":<10}|{"":<10}|{"":<10}|{test_acc.mean():<10.4f}|{"":<10}|{test_loss.mean():<10.4f}')

    return get_params, best_opt_state, (hist_train_loss, hist_val_loss, hist_test_loss)
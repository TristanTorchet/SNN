class SimArgs:
    def __init__(self, n_in: int, n_h: int, seed: int, tau_mem: float, transition_steps: int, transition_begin: int,
                 noise_std: float, nb_epochs: int):
        # archi
        self.n_in = n_in
        self.n_h = n_h
        self.n_out = 20
        # weight
        self.w_scale = 0.3
        self.pos_w = True # use only positive weights at initizialization
        self.noise_std = noise_std  # [0.05, 0.1, 0.15, 0.2]
        # neuron model
        self.tau_mem = tau_mem
        self.v_thr = 1
        # data
        self.nb_rep = 1
        self.timestep = 0.005 # 280 timesteps
        self.truncation = True # to use only 150 of 280 timesteps
        # training
        self.lr = 0.001
        self.nb_epochs = nb_epochs
        self.batch_size = 64
        self.seed = seed
        # self.lr_config = 2
        self.transition_steps = transition_steps
        self.transition_begin = transition_begin


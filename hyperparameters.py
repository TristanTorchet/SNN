class SimArgs:
    def __init__(self, n_in: int, n_h: int, bias_enable: bool, train_tau: bool, seed: int, tau_mem: float, tau_syn: float, nb_epochs: int, lr: float):
        # archi
        self.n_in = n_in
        self.n_h = n_h
        self.n_out = 20
        self.bias_enable = bias_enable
        # weight
        self.w_scale = 0.3
        self.pos_w = True # use only positive weights at initizialization
        # neuron model
        self.tau_mem = tau_mem
        self.v_thr = 1
        self.tau_syn = tau_syn
        self.train_tau = train_tau
        # data
        self.nb_rep = 1
        self.timestep = 0.005 # 280 timesteps
        self.truncation = True # to use only 150 of 280 timesteps
        # training
        self.lr = lr
        self.nb_epochs = nb_epochs
        self.batch_size = 64
        self.seed = seed

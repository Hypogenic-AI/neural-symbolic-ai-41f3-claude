"""Central experiment configuration (hyperparameters and grids)."""

# Shared training hyperparameters
EPOCHS = 12
BATCH_SIZE = 128
LR = 1e-3
SEEDS = [0, 1, 2, 3, 4]            # 5 seeds for statistics

# E1/E2: data-efficiency grid (number of training pairs)
TRAIN_SIZES = [100, 500, 1000, 5000, 15000, 30000]
MODELS = ["neural", "nesy", "oracle"]

# E3: mitigations evaluated at a low-data regime where the grounding gap is largest
E3_TRAIN_SIZE = 1000
E3_MITIGATIONS = [
    {"name": "baseline", "aux_digit_k": 0, "uniform_prior": 0.0, "entropy_reg": 0.0},
    {"name": "aux_k1",   "aux_digit_k": 1, "uniform_prior": 0.0, "entropy_reg": 0.0},
    {"name": "aux_k5",   "aux_digit_k": 5, "uniform_prior": 0.0, "entropy_reg": 0.0},
    {"name": "aux_k20",  "aux_digit_k": 20, "uniform_prior": 0.0, "entropy_reg": 0.0},
    {"name": "uniform_prior", "aux_digit_k": 0, "uniform_prior": 1.0, "entropy_reg": 0.0},
    {"name": "entropy_reg",   "aux_digit_k": 0, "uniform_prior": 0.0, "entropy_reg": 0.1},
]

# E5: shortcut induction via modular addition (sum mod 10)
E5_TRAIN_SIZES = [1000, 5000, 30000]
E5_MIT_SIZE = 5000               # mitigation sweep size under modular addition

# E4: compositional OOD
E4_TRAIN_SIZE = 30000             # use the full single-digit training set
E4_N_DIGITS = [2, 3]              # test 2- and 3-digit number addition
E4_N_PAIRS = 5000

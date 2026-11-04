"""Frozen Stage-B K8 codec warmup recipe; training is not launched by importing this."""
UPDATES = 10_000
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP_NORM = 1.0
VALIDATION_EVERY = 250
VALIDATION_BATCHES = 4
SAVE_EVERY = 250
CDL_MODEL = "C"
SPEED_MPS = 1.0
BASE_DATA_SEED = 20261106
SEED_NAMESPACE = "twc_v1_k8_codec_warmup_v1"
CE_CHECKPOINT = "runs/twc_v1_k8_ce/seed_20261105/checkpoint_latest.pt"
CE_CALIBRATION = "runs/twc_v1_k8_ce/seed_20261105/calibration.csv"
PILOT_ENERGIES = (8.0, 16.0, 24.0)
ENERGY_RATIOS = ((1, 3), (1, 1))
GAMMAS = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0)

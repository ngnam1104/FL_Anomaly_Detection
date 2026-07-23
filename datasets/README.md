# SMAP and MSL benchmark data

The Ubuntu experiment matrix uses the public Telemanom SMAP/MSL package,
dataset version 1. Prepare it with:

```bash
.venv/bin/python -m scripts.prepare_benchmarks --dataset nasa
```

`run_ubuntu.sh` performs this step automatically. The extracted layout is:

```text
datasets/
├── _downloads/
│   └── nasa_smap_msl.zip
└── telemanom/
    ├── labeled_anomalies.csv
    ├── train/
    │   └── <channel>.npy
    └── test/
        └── <channel>.npy
```

Both the archive and extracted benchmark are ignored by Git.

## Mapping source channels to 100 sensors

SMAP contains 55 source channels with feature dimension 25. MSL contains 27
source channels with feature dimension 55. Every experiment nevertheless uses
the same physical topology:

```text
N = 100 stationary sensors
M = 10 mobile fog/AUV aggregators
```

For each source channel, the first 90% of its normal training sequence is used
for local training and the remaining 10% for normal-only threshold validation.
The local-training sequences are split into 100 contiguous, non-overlapping
sensor shards, allocated approximately in proportion to sequence length.

This mapping does not duplicate samples and does not mix samples from different
source channels inside one sensor. Test sequences and inclusive anomaly ranges
remain unchanged and are only used for evaluation.

SMAP and MSL are trained and evaluated independently because their input
dimensions differ. “Cross-dataset” here means applying the same topology,
baselines and hyperparameters to both benchmarks, not training on SMAP and
directly testing the same autoencoder on MSL.

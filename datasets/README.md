# Benchmark datasets

Prepare all three public benchmarks on Windows:

```powershell
python -m scripts.prepare_benchmarks
```

The command downloads:

- SMD from the OmniAnomaly `ServerMachineDataset` directory, pinned to commit
  `7fb0e0acf89ea49908896bcc9f9e80fcfff6baf4`.
- SMAP/MSL from the public Telemanom Kaggle package, dataset version 1.

Downloaded archives stay under ignored `datasets/_downloads/`. Prepared data:

```text
datasets/
├── SMD/
│   ├── train/*.txt
│   ├── test/*.txt
│   └── test_label/*.txt
└── telemanom/
    ├── train/*.npy
    ├── test/*.npy
    └── labeled_anomalies.csv
```

The paper configuration uses the first 10 SMD machines. SMAP keeps all 55
channels as 55 FL clients with `D=25`; MSL keeps all 27 channels as 27 clients
with `D=55`. Telemanom anomaly ranges are inclusive at both endpoints.

Sources:

- https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset
- https://github.com/khundman/telemanom
- https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl

## Alternative processed layout

Real benchmarks are intentionally not committed. Place processed arrays here:

```text
<NAME>_train.npy
<NAME>_test.npy
<NAME>_test_label.npy
```

where `<NAME>` is `SMD`, `SMAP`, or `MSL`. The alternative
`<NAME>/{train.npy,test.npy,labels.npy}` layout is also supported.

For raw SMD, use `SMD/{train,test,test_label}/*.txt`.

# Data

The raw MHEALTH dataset is **not committed** to the repository (it is ~1.2 M rows
across 10 subjects). Fetch it with:

```bash
python scripts/download_data.py
```

This downloads, verifies, and extracts the dataset from the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/319/mhealth+dataset)
into `data/raw/MHEALTHDATASET/`. The pinned archive is 75,567,983 bytes with SHA-256
`16ad0ce709f3f00df18f348610d15bce0884b79e2143f57f446493673f02b8e0`.
The archive and extracted file hashes are recorded locally by the download command.

```
data/raw/MHEALTHDATASET/
├── mHealth_subject1.log
├── ...
├── mHealth_subject10.log
├── README.txt
└── .dataset-provenance.json
```

Each `.log` file is whitespace-separated with 24 columns: 23 body-worn sensor
channels (accelerometer, gyroscope, magnetometer at the chest, ankle and arm,
plus 2-lead ECG) sampled at **50 Hz**, and a final activity label in `0..12`
(`0` is the null/transition class and is dropped during feature building).

## Citation

The UCI dataset is [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and has
[DOI 10.24432/C5TW22](https://doi.org/10.24432/C5TW22). Keep the downloaded traces out of version control.

> Banos, O., Garcia, R., Holgado-Terriza, J.A., Damas, M., Pomares, H., Rojas, I.,
> Saez, A., Villalonga, C. *mHealthDroid: a novel framework for agile development
> of mobile health applications.* IWAAL 2014.

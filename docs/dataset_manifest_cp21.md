# CP21 dataset manifest — UCI Iris and Bike Sharing

This manifest records the approved sources and the transformations applied by
the CP21 pipeline. Both original datasets are distributed by the UCI Machine
Learning Repository under CC BY 4.0. Preserve the cited attribution whenever
the datasets or derived data are redistributed.

## UCI Iris — classification

- **UCI dataset ID:** 53
- **Task:** Three-class classification.
- **Source archive:** https://archive.ics.uci.edu/static/public/53/iris.zip
- **Dataset citation:** Fisher, R. A. (1936). Iris. UCI Machine Learning
  Repository. DOI: https://doi.org/10.24432/C56C76
- **Attribution:** R. A. Fisher; dataset distributed by the UCI Machine
  Learning Repository.
- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0),
  as listed by UCI.
- **Source member:** iris.data.
- **Features:** sepal length, sepal width, petal length and petal width, in
  centimetres.
- **Target mapping:** Iris-setosa -> 0, Iris-versicolor -> 1,
  Iris-virginica -> 2.
- **Integrity checks:** exactly 150 observations, 50 per class, five fields
  per row, finite measurements, and only the three documented labels.
- **Split:** seeded, stratified 70%/15%/15% train/validation/test. The
  deterministic largest-remainder allocation yields 105/23/22 observations
  at the configured fractions; every class is represented in each split.
- **Scaling:** population mean and standard deviation are fitted on the
  training split only, then reused for validation and test. A constant
  training feature receives scale 1 to avoid division by zero.

## UCI Bike Sharing — hourly regression

- **UCI dataset ID:** 275
- **Task:** predict total hourly rental count, cnt.
- **Source archive:**
  https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip
- **Dataset citation:** Fanaee-T, H. (2013). Bike Sharing. UCI Machine
  Learning Repository. DOI: https://doi.org/10.24432/C5W894
- **Attribution:** Hadi Fanaee-T; dataset distributed by the UCI Machine
  Learning Repository.
- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0),
  as listed by UCI.
- **Source member:** hour.csv; the daily table is not used.
- **Features:** season, yr, mnth, hr, holiday, weekday, workingday,
  weathersit, temp, atemp, hum, and windspeed.
- **Target:** cnt, retained in its original bike-rental count units.
- **Excluded columns:** casual and registered are direct components of
  cnt (cnt = casual + registered) and therefore leak target information.
  instant is a record identifier and dteday is used only to construct
  temporal splits; neither is a model input.
- **Integrity checks:** required columns, valid ISO dates, non-decreasing
  chronology, finite model features, non-negative integer counts, and the
  casual + registered = cnt identity on every row.
- **Split:** chronological by unique calendar day. The earliest 70% of days
  are training, the next 15% validation, and the remaining days test. All
  rows from a calendar day remain in the same split. Date ranges are saved
  in metadata. No random shuffling is applied to this time-series split.
- **Scaling:** fit on training rows only and reused for validation/test.
  Constant training features receive scale 1.

## Storage and reproducibility

- Configuration: configs/cp21_dataset_pipeline.json; seed 42.
- Raw downloads: data/raw/.
- Prepared numerical arrays: data/processed/*.npz.
- Provenance: one JSON sidecar per prepared archive records source URL,
  SHA-256 digest of the exact downloaded archive, dataset ID/DOI, licence,
  attribution, feature/target names, split information and array shapes.
- The existing .gitignore excludes data/; neither raw third-party rows
  nor generated model artefacts are committed.
- Download code accepts HTTPS only from the exact archive.ics.uci.edu
  hostname, applies size/time limits, checks redirects and never extracts
  ZIP paths to disk.
- Prepared outputs refuse to overwrite earlier results. Use a fresh output
  path when intentionally generating another dataset version.

Run these commands from the repository root:

    python -m pytest -q tests/test_datasets.py
    python -m experiments.cp21_dataset_pipeline

The pipeline's actual source archive checksums and exact Bike Sharing split
row counts must be captured by a successful run with network access. The
repository report does not claim those run-time values before they exist.

"""Multi-Model Benchmarking on the MHEALTH wearable-sensor dataset.

Public modules
--------------
- :mod:`src.data_loader`   -- load raw ``.log`` files and build windowed features
- :mod:`src.preprocessing` -- train/test split and standardisation
- :mod:`src.metrics`       -- accuracy, F1, ROC-AUC (from scratch)
- :mod:`src.models`        -- KNN, Logistic Regression, Naive Bayes, Neural Net
"""

__version__ = "1.0.0"

# Shared feature and scoring code

core.py validates inputs and scores frozen pipelines. history.py computes strictly
earlier per-card features. shared/project_data.py contains common transaction
preparation. model_sharing.py verifies trusted model files and example predictions.

Training is in Part_III/Modeling; the app is in Final_Demo. Both import these same
modules. project_paths.py at repository root locates the project independently of
the working directory and preserves the module names stored in joblib pipelines.
Datasets and models do not belong in Shared. The algorithms were not changed by
the directory reorganization.

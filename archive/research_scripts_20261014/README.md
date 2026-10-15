# Archived research scripts

These files were removed from the active src folder after Tuned 80 became the sole active prediction model. Each file is preserved byte for byte; manifest.json records the original path, checksum and reason.

- 11_train_40_feature_lightgbm.py: training and tuning the Tuned 40 comparison model.
- 09_calibrate_lightgbm.py: calibration for the Original 80 model, not Tuned 80.
- 12_rdap_controlled_audit.py: completed historical RDAP counterfactual audit.
- 08_test_domain_backup.py: old duplicate prediction script.

The active application uses src/08_test_domain.py and src/14_batch_test_domains.py, with lightgbm_80_tuned_model.joblib and training_preprocessor_80_tuned.joblib. Dataset preparation, training, evaluation, feature analysis and Tuned 80 tuning remain under src.

To reproduce an archived experiment, first copy its file back to the original src path recorded in manifest.json. Check that the destination does not already contain a newer file. The scripts derive the project root from their original location; running them directly inside this archive would use incorrect data/model paths.

Models, datasets, previous evaluation results and tests were preserved.

# TODO - Improve model scoring / regression metrics

- [x] Confirm issue: current pipeline is classification (uses `accuracy_score`) while target `Grade` is cast to string.
- [x] Update `src/components/data_transformation.py` to treat `Grade` as regression (convert to numeric) and keep y as numeric.
- [x] Update `src/components/model_trainer.py` to use regressors and print MAE, MSE, RMSE, R2 + cross-validation.
- [x] Update `app.py` to display predicted grade as numeric.

- [x] Run `src/pipeline/prediction_pipeline.py` and/or training to verify different metrics are printed.



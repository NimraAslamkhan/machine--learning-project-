- [ ] Fix prediction crash caused by sklearn version mismatch in persisted artifacts
  - [ ] Pin scikit-learn version in requirements.txt to match artifact training (1.7.2)
  - [ ] Remove/ignore duplicate or outdated PredictPipeline implementation in src/pipeline/training_pipeline.py (keep prediction_pipeline.py version)
  - [ ] Ensure Streamlit app uses the corrected PredictPipeline
  - [ ] Quick smoke test: run app prediction path and confirm no _fill_dtype error


# LOBnet Codebase

Repository containing several deep learning models enabling prediction of stock price movements using Limit Order Book (LOB) data. It includes our enhanced DeepLOB Network with improved regularization and a novel hybrid CNN-Transformer architecture, tested on the FI-2010 dataset to explore trade-offs between accuracy, complexity, and robustness in financial forecasting.

## Running the models

To run the models locally, you will first need to download the FI-2010 dataset locally and convert the .txt files to .csv:

1. Download `BenchmarkDatasets.zip` from https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649/data
2. Unzip the file and move the resulting `BenchmarkDatasets` directory to the `data` directory at the root of the project
3. Run the script to convert the dataset to CSVs: `python3 scripts/convert_data_to_csv.py`. Ensure that the root path is correctly pointed at the root path in the unzipped folder.
4. From there, you can run our models (found in `LOBnet/final_models.py`) against the test dataset through `notebooks/model_evaluation` or train the individual models in `notebooks/train_models`

## Contribution

- Rafael Singer: transformer model notebook, transformer model in `final_models.py`, architecture diagrams, helped with `orderbook.py`, refactored and cleaned up a lot of the code
- Jack Klingenberg: `Orderbook.py`, data download script, csv conversion script, DeepLOB paper & v2 models in `final_models.py`, helped with model training notebook (accuracy curves), trained models.
- Tommy DeNezza: `model_evaluation.py`, `train_models.py`, `baseline_model.py`, precursor models to v2, creating infrastructure for saving models

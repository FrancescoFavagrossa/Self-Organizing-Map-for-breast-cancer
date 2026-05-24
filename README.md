# Self-Organizing Map for Breast Cancer Data

This repository demonstrates a Self-Organizing Map (SOM) trained on the Wisconsin breast cancer dataset from `scikit-learn`.

The main notebook trains a 12 x 12 SOM, visualizes the final U-matrix, maps benign and malignant samples onto the learned grid, and exports an animated GIF showing how the SOM evolves during training.

![SOM training animation](som_breast_cancer_animation.gif)

## Files

- `SOM_and_Breast_cancer.ipynb` - main notebook for loading data, training the SOM, visualizing results, and generating the animation.
- `som_breast_cancer_animation.gif` - exported animation of the SOM U-matrix during training.
- `som_test.py` - small MiniSom test script using the Iris dataset.
- `Untitled-1.jl` - standalone Julia SOM experiment.

## Dataset

The project uses the built-in breast cancer dataset from `sklearn.datasets.load_breast_cancer`.

It contains:

- 569 samples
- 30 numeric features
- 2 diagnosis classes: `benign` and `malignant`

The features are scaled with `MinMaxScaler` before SOM training.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install numpy pandas matplotlib scikit-learn minisom pillow jupyter
```

## Run The Notebook

Start Jupyter:

```bash
jupyter notebook
```

Open:

```text
SOM_and_Breast_cancer.ipynb
```

Run all cells. The notebook will:

1. Load the breast cancer dataset.
2. Scale the feature matrix.
3. Train a Self-Organizing Map.
4. Plot the final U-matrix.
5. Plot sample locations by diagnosis.
6. Save the SOM animation as `som_breast_cancer_animation.gif`.

## Output

The animation shows the SOM U-matrix changing over training iterations. Lighter and darker regions represent differences in average distance between neighboring SOM nodes, helping reveal cluster structure in the dataset.

## Notes

The SOM is trained in an explicit loop instead of a single `train_random` call so that intermediate U-matrix snapshots can be captured for animation.


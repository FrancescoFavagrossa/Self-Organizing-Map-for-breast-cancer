 # Self-Organizing Map for Breast Cancer Data

This repository demonstrates a Self-Organizing Map (SOM) trained on the Wisconsin breast cancer dataset from `scikit-learn`.
This project uses a Self-Organizing Map (SOM) to explore the Wisconsin breast cancer dataset from `scikit-learn`.

The main notebook trains a 12 x 12 SOM, visualizes the final U-matrix, maps benign and malignant samples onto the learned grid, and exports an animated GIF showing how the SOM evolves during training.
The notebook trains a 12 x 12 SOM, visualizes the final U-matrix, maps benign and malignant samples onto the grid, and exports an animated GIF of the learning process.

<img width="1400" height="1400" alt="som_breast_cancer_animation" src="https://github.com/user-attachments/assets/b4df4323-a554-4e0b-9f66-454a2f0c00e9" />

## What Is A SOM?

A Self-Organizing Map is an unsupervised neural network used for dimensionality reduction and clustering. It projects high-dimensional data onto a usually 2D grid while trying to preserve neighborhood relationships.

Each map node has a weight vector:

$$
\mathbf{w}_i \in \mathbb{R}^d
$$

where `d` is the number of input features. For this dataset, `d = 30`.

For an input sample:

$$
\mathbf{x} \in \mathbb{R}^d
$$

the closest map node is selected as the Best Matching Unit (BMU):

$$
c = \arg\min_i \lVert \mathbf{x} - \mathbf{w}_i \rVert
$$

The BMU and its neighbors are then moved closer to the input sample:

$$
\mathbf{w}_i(t + 1) =
\mathbf{w}_i(t) +
\alpha(t) h_{ci}(t)
\left(\mathbf{x}(t) - \mathbf{w}_i(t)\right)
$$

where:

- $\alpha(t)$ is the learning rate.
- $h_{ci}(t)$ is the neighborhood function between the BMU `c` and node `i`.

A common neighborhood function is Gaussian:

$$
h_{ci}(t) =
\exp\left(
-\frac{\lVert \mathbf{r}_c - \mathbf{r}_i \rVert^2}
{2\sigma(t)^2}
\right)
$$

where $\mathbf{r}_c$ and $\mathbf{r}_i$ are positions on the 2D SOM grid, and $\sigma(t)$ controls the neighborhood radius.

Early in training, $\sigma(t)$ is large, so broad structure forms across the map. Later, $\sigma(t)$ becomes smaller, allowing local details to settle.

The result is a map where similar samples tend to appear near each other. In this project, the SOM helps show how benign and malignant breast cancer samples organize across the learned grid.

The U-matrix visualizes distances between neighboring SOM nodes. Darker or lighter regions can reveal cluster boundaries, depending on the colormap.

## Dataset

The dataset contains 569 samples, 30 numeric features, and two diagnosis labels: `benign` and `malignant`. Features are scaled with `MinMaxScaler` before training.

## Main Files

- `SOM_and_Breast_cancer.ipynb` - main notebook.
- `som_breast_cancer_animation.gif` - exported SOM training animation.
- `som_test.py` - small MiniSom test with the Iris dataset.
- `Untitled-1.jl` - standalone Julia SOM experiment.

## How To Run

Install the Python dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn minisom pillow jupyter
```

Then start Jupyter and open `SOM_and_Breast_cancer.ipynb`:

```bash
jupyter notebook
```

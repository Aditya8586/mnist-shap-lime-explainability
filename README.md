# MNIST Digit Classifier with SHAP & LIME Explanations

1. **Model** – A fully connected neural network built with TensorFlow/Keras (Flatten → Dense 128 ReLU → Dense 64 ReLU → Dense 10 softmax) that classifies 28×28 grayscale handwritten digits from the MNIST dataset. Pixel values are normalized to [0, 1] and the model is trained with Adam and sparse categorical cross-entropy, reaching over 99% training accuracy within a few epochs.

2. **Explainability** – Predictions are interpreted with two post-hoc XAI methods: **SHAP** (`DeepExplainer`, 100-sample background set) shows per-pixel contributions for the first five test images, and **LIME** (`LimeImageExplainer`, 1,000 perturbation samples) highlights the image regions that most support the predicted digit.

3. **Usage** – Install the dependencies with `pip install numpy tensorflow matplotlib shap lime scikit-image`, then run `MNIST.ipynb` top to bottom in Jupyter. The MNIST data downloads automatically via `keras.datasets`.

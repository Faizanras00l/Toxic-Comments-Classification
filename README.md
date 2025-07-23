# Toxic Comment Classification

This project contains code and resources for training and evaluating a multi-label toxic comment classifier using PyTorch and scikit-learn. The main workflow is provided in a Jupyter Notebook (`Toxic_Comment.ipynb`). This project can be run locally or in Google Colab.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Running the Notebook](#running-the-notebook)
- [Uploading Model Weights and Artifacts](#uploading-model-weights-and-artifacts)
- [Paths to Update](#paths-to-update)
- [Usage Example](#usage-example)
- [License](#license)

---

## Project Structure

```
.
├── Toxic_Comment.ipynb
├── train.csv
├── test.csv
├── test_labels.csv
├── weights/
│   ├── toxic_model_weights.pth
│   ├── optimizer_state.pth
│   ├── vectorizer.pkl
│   └── model_info.pkl
├── metrics/
│   ├── metrics.csv
│   └── *.png
└── README.md
```

---

## Setup Instructions

1. **Download or copy the project files** to your working directory.

2. **Install dependencies:**

    - **Locally:**  
      Install Python 3.8+ and run:
      ```sh
      pip install torch scikit-learn pandas matplotlib
      ```

    - **Google Colab:**  
      The notebook will install dependencies automatically.

---

## Running the Notebook

1. **Open `Toxic_Comment.ipynb`**  
   - In Jupyter Notebook, JupyterLab, or upload to [Google Colab](https://colab.research.google.com/).

2. **Upload Data Files**  
   - Upload `train.csv`, `test.csv`, and `test_labels.csv` to the notebook’s working directory.

3. **Upload Model Weights and Artifacts**  
   - If you want to use a pre-trained model (for inference or evaluation), upload the following files to a folder named `weights/` in the notebook’s working directory:
     - `toxic_model_weights.pth`
     - `optimizer_state.pth`
     - `vectorizer.pkl`
     - `model_info.pkl`

   - If these files do not exist, you can run the training cells to generate them.

---

## Uploading Model Weights and Artifacts

**Where to upload:**

- Create a folder named `weights` in the same directory as your notebook.
- Upload the following files into `weights/`:
    - `toxic_model_weights.pth`
    - `optimizer_state.pth`
    - `vectorizer.pkl`
    - `model_info.pkl`

**How to update paths in the notebook:**

- In the inference section (usually the last code cell), make sure the file paths match:
    ```python
    with open("weights/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("weights/model_info.pkl", "rb") as f:
        model_info = pickle.load(f)
    model.load_state_dict(torch.load("weights/toxic_model_weights.pth", map_location=device))
    ```

- If you upload the files elsewhere, update the paths accordingly.

---

## Usage Example

After setup, you can run the inference cell to classify new comments:

```python
sample_comments = [
    "You are so stupid!",
    "Have a nice day!"
]
preds = predict_toxic(sample_comments)
for comment, pred in zip(sample_comments, preds):
    print(f"Comment: {comment}")
    print(explain_prediction(comment, pred))
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

**Note:**  
- If you use Google Colab, use the file upload button (`Files` sidebar or `from google.colab import files`) to upload weights and data files.
- Always ensure the paths in the notebook match the location of your uploaded files.
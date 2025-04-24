
# DAT550 - Document Classification using Neural Networks

This project explores various techniques for text classification using the Arxiv dataset. It evaluates different embedding strategies and model architectures, ranging from traditional TF-IDF with MLP to Word Embedding-based FFNN and RNNs.

---

## 📁 Dataset

The dataset is sourced from:
- `/content/drive/MyDrive/DAT550/arxiv_train.csv.gz`
- `/content/drive/MyDrive/DAT550/arxiv_test.csv.gz`

Each record includes:
- `abstract`: A scientific paper's abstract
- `label`: The category of the paper

---

## ⚙️ Environment Setup
We recommend using Google Colab in combination with Google Drive to view and run this notebook.
The models are trained and tested using Google Colab, while the datasets are accessed from Google Drive.
Before running the code, make sure to download and upload all the necessary files to your Drive.
Also, ensure the following packages are installed in the notebook:

```bash
pip install gensim
pip install numpy==1.23.5
pip install --force-reinstall --no-cache-dir gensim
```
While intalling these packages, you'll see an error to restart the session, make sure to wait for the installation get completed and then restart the notebook.

Also ensure access to Google Drive is available for dataset loading:
```python
from google.colab import drive
drive.mount('/content/drive')
```

All the above code is included in the file. 

---

## 📥 Pre-trained Embeddings Required

Before running the notebook, download and place the following embeddings into your Google Drive:

1. **GloVe (100-dimensional)**  
   - File: `glove.6B.100d.txt`
   - [Download Link](https://nlp.stanford.edu/data/glove.6B.zip)

2. **FastText (English, 300-dimensional or similar)**  
   - File: `wiki-news-300d-1M.vec`
   - [Download Link](https://fasttext.cc/docs/en/english-vectors.html)

3. **Word2Vec (Google News, 300-dimensional)**  
   - File: `GoogleNews-vectors-negative300.bin.gz`
   - [Download Link](https://code.google.com/archive/p/word2vec/)

📝 **Place these files in a known location in your Google Drive**, such as:
```
/content/drive/MyDrive/DAT550/
```

Then, update the paths in your notebook accordingly to load them.

---

## 📊 Approach Overview

### 1. **TF-IDF + MLP**
- Convert abstracts into TF-IDF vectors using `TfidfVectorizer` (max_features = 20,000).
- Encode labels using `LabelEncoder`.
- Train an `MLPClassifier` with one hidden layer (128 neurons).
- Evaluate performance using accuracy and classification report.

Contributor: **Rabia Zakriyya**

---

### 2. **Word Embeddings + Feedforward NN**
- Use pre-trained embeddings:
  - Word2Vec
  - GloVe
  - FastText
- Aggregate word vectors for each abstract using:
  - Mean
  - Max
  - Sum
- Feed aggregated vectors to a simple Feedforward Neural Network.
- Compare accuracy across all embeddings and strategies.

Contributors: **Nabeel Raja, Arbaz Ali**

---

### 3. **Word Embeddings + Recurrent Neural Networks (RNNs)**
- Use the same embeddings (Word2Vec, GloVe, FastText).
- Test with both GRU and LSTM architectures.
- Try different pooling strategies (mean, max, sum).
- Evaluate performance on test data.

Contributors: **Daud Sadiq**

---

## 📈 Visualizations

Generated graphs to compare:
- Test accuracy for different aggregation strategies.
- RNN type vs test accuracy (per aggregation strategy).

---

## 📌 Summary

- TF-IDF + MLP provides a solid baseline.
- Word embedding methods improve semantic representation.
- GRU and LSTM enhance sequential understanding.
- Pooling strategies significantly influence final performance.

---

## 🔬 Authors and Credits

**Course: DAT550**

Institution: **University of Stavanger**

Contributors: **Daud Sadiq, Arbaz Ali, Nabeel Raja, Rabia Zakriyya**


# python-ml-by-example-exercises
Worked examples and exercises from Python Machine Learning By Example (4th Edition) by Yuxi Liu, with additional tweaks and enhancements, including custom summarization features.
Hands-on implementations and exercises from *Python Machine Learning By Example (4th Ed.)*, with personal enhancements and portfolio projects.

![Python](https://img.shields.io/badge/python-3.10-blue.svg) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

This repository contains my practical work, implementations, and summaries developed while studying the book **Python Machine Learning By Example, Fourth Edition** (July 2024) by Yuxi (Hayden) Liu.

The goal of this project is to provide a hands-on portfolio demonstrating proficiency across major Machine Learning (ML) paradigms, including Supervised Learning, Unsupervised Learning, Deep Learning (DL), and Reinforcement Learning (RL). The notebooks emphasize modern ML best practices and techniques such as feature engineering, model tuning, and regularization.

## 🧠 Core Features and Highlights

Covers the end-to-end ML workflow: data preprocessing → modeling → evaluation → deployment best practices (MLOps).

### Supervised Learning (Classification & Regression)

* **Naïve Bayes** (Chapter 2): Movie recommendation engine, prior probabilities, Laplace smoothing, AUC/ROC, confusion matrix.
* **Tree-Based Models** (Chapter 3): Ad click-through prediction using Decision Trees, Random Forests, GBT; Gini vs. Information Gain.
* **Logistic Regression** (Chapter 4): SGD-based implementation, one-hot encoding, L1/L2 regularization.
* **Traditional Regression** (Chapter 5): Stock forecasting with Linear Regression and tree regressors; MSE/RMSE/R².
* **Support Vector Machine (SVM)** (Chapter 9): Multi-class classification, kernels (RBF, poly), soft margins, face recognition project.

### Deep Learning (DL)

* **ANNs** (Ch. 6, 11): Architecture, activations, backprop, dropout, early stopping.
* **CNNs** (Ch. 11): Fashion-MNIST, conv/pool layers, data augmentation, transfer learning.
* **RNNs / LSTM** (Ch. 12): Sequential models — sentiment analysis, text generation.
* **Transformers** (Ch. 13): Self-attention, BERT/GPT concepts for modern NLP.
* **Multimodal (CLIP)** (Ch. 14): Vision-language demonstration (text-based image search).

### Unsupervised Learning & RL

* **Text Analysis / Dimensionality Reduction** (Ch. 7): Tokenization, stemming, t-SNE, word embeddings.
* **Clustering & Topic Modeling** (Ch. 8): K-means, Elbow method, NMF, LDA.
* **Reinforcement Learning** (Ch. 15): DP (value/policy iteration), Q-Learning fundamentals.

---

## 🛠️ Requirements & Setup

**Language:** Python 3.x
**Core libraries:** `numpy`, `pandas`, `scikit-learn`
**Deep Learning:** `tensorflow`/`keras`, `torch` (for advanced examples)
**NLP/Advanced:** `transformers`, `nltk`, `spacy`, `gymnasium` (for RL)

Create a virtual environment and install dependencies:

```bash
git clone https://github.com/your-username/python-ml-by-example-exercises.git
cd python-ml-by-example-exercises
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Open notebooks:

```bash
jupyter notebook
# or open in Colab using the "Open In Colab" badge above
```

---

## 📁 Repository Structure (overview)

```
Colab Notebooks/
│
├─ chapter_2/
│  ├─ betterImplementation.ipynb
│  ├─ Exercice1.ipynb
│  ├─ Exercice2.ipynb
│  ├─ movie_recommendation.ipynb
│  └─ naive-bayes.ipynb
│
├─ chapter_3/
│  ├─ PART1.ipynb
│  └─ PART2.ipynb
│
├─ chapter_4/
│  ├─ SGD-based logistic regression algorithm from scratch.ipynb
│  ├─ SGD-based logistic regression algorithm_using_tools.ipynb.ipynb
│  └─ understanding_LR_and_one-of-K.ipynb
│
├─ chapter_5/
│  ├─ Exercice.ipynb
│  └─ official.ipynb
│
├─ chapter_6/
│  ├─ Exercices.ipynb
│  └─ official.ipynb
│
├─ chapter_7/
│  ├─ Exercices.ipynb
│  └─ locala.ipynb
│
├─ chapter_8/
│  ├─ Exercice.ipynb
│  └─ official.ipynb
│
├─ chapter_9/
│  ├─ Exercices.ipynb
│  └─ official.ipynb
│
├─ chapter_10/
│  └─ Best_Practice.ipynb
│
├─ chapter_11/
│  └─ official.ipynb
│
├─ chapter_12/
│  └─ Untitled0.ipynb
│
├─ Download_ad click-through.ipynb
├─ foodClassifier.ipynb
├─ important_terms.txt
├─ italic_words.txt
├─ sckitlearn.ipynb
├─ summary_technical_CNN_&_RNN.txt
├─ summary_technical_supervised.txt
└─ summary_technical_unsupervised.txt
```

### Auxiliary files

* `Download_ad click-through.ipynb` — data download & preprocessing utility for ad click-through dataset.
* `foodClassifier.ipynb` — classification exercise (SVM / tree based).
* `sckitlearn.ipynb` — scikit-learn usage examples (preprocessing, pipeline, estimators).
* `important_terms.txt`, `italic_words.txt` — quick glossaries.
* `summary_technical_*.txt` — compact technical notes & code snippets for each domain.

---

## 🚀 Quick Start / Minimal Example

1. Open `Colab Notebooks/chapter_11/official.ipynb` in Colab to run the CNN example (Fashion-MNIST).
2. For local runs of smaller notebooks: install `requirements.txt`, then `jupyter notebook` and open the chosen file.
3. For large notebooks (chapter_6 official, chapter_7 Exercices) prefer Colab or a machine with >8GB RAM.

---

## 🤝 Contributing & Notes

This repository documents my work on the book exercises and my personal enhancements. Feedback, small fixes, or suggested improvements are welcome — particularly:

* Optimizations and refactors of large notebooks
* Additional visualizations or smaller reproducible scripts
* Updated techniques or more efficient implementations

If you contribute, please open a PR and include a brief description of changes.

---

## 📚 License & Book Copyright

This repo is for learning and demonstration. Respect the book's copyright and licensing — do not redistribute the book content itself. Code is provided for educational use. Add a LICENSE file as needed (e.g., MIT) to clarify reuse.

---

## 📝 Suggestions & Improvements (reviewer notes)

> Your README is already very thorough and professional! It’s clear, detailed, and presents your work impressively. That said, I have some **practical suggestions and improvements** to make it even more polished, readable, and GitHub-friendly:
>
> ---
>
> ### **1. Add a concise project tagline at the top**
>
> Right now, your title is long. Add a one-line summary under it so readers immediately know what this repo is.
> Example:
>
> ```markdown
> Hands-on implementations and exercises from *Python Machine Learning By Example (4th Ed.)*, with personal enhancements and portfolio projects.
> ```
>
> ---
>
> ### **2. Simplify long tables or break them into sections**
>
> Your tables for DL, RL, and ML algorithms are excellent, but they are very long. Consider splitting them by **topic type** or **chapter range**, or using bullet points for readability. GitHub renders big tables slowly.
>
> Example (bullet version for SVM instead of a big table row):
>
> ```markdown
> - **Support Vector Machine (SVM)** (Chapter 9)  
>   Multi-class classification for face recognition. Concepts: decision hyperplane, support vectors, soft margins, kernel methods (RBF, Polynomial).
> ```
>
> ---
>
> ### **3. Use emojis judiciously**
>
> You already have some (`🧠`, `🛠️`, `📁`), which is great for visual cues. You could also add:
>
> * 📊 for tables,
> * 📘 for chapters,
> * ⚡ for tips or highlights.
>
> ---
>
> ### **4. Clarify “Auxiliary Files”**
>
> Right now, `sckitlearn.ipynb` and some summaries might confuse readers. Add whether they are exercises, experiments, or your personal notes.
>
> ---
>
> ### **5. Add a “Getting Started / Quick Run” Section**
>
> Many GitHub visitors look for immediate instructions. A minimal example:
>
> ````markdown
> ## 🚀 Getting Started
>
> 1. Clone the repo:
> ```bash
> git clone https://github.com/your-username/python-ml-by-example-exercises.git
> cd python-ml-by-example-exercises
> ````
>
> 2. Install dependencies:
>
> ```bash
> pip install -r requirements.txt
> ```
>
> 3. Open any notebook in `Colab Notebooks/` or run scripts locally:
>
> ```bash
> jupyter notebook
> ```
>
> ````
>
> ---
>
> ### **6. Add a “Contributing / Notes” Section**
> Even if it’s your personal portfolio, a short note like this improves professionalism:
>
> ```markdown
> ## 🤝 Contributing
>
> This repository showcases my work from the book exercises and personal enhancements. Contributions, improvements, or discussions are welcome, especially regarding:
> - Optimizations of existing notebooks
> - Additional visualizations or datasets
> - Updated techniques from modern ML/DL research
> ````
>
> ---
>
> ### **7. Optional: Add badges**
>
> Badges improve repo visibility. Example badges for Python version, Colab, or license:
>
> ```markdown
> ![Python](https://img.shields.io/badge/python-3.10-blue.svg)
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](link-to-notebook)
> ```

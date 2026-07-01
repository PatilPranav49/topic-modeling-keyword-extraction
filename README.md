# Topic Modeling and Keyword Extraction using Latent Dirichlet Allocation (LDA)

A Streamlit-based Natural Language Processing application that performs topic modeling on textual data using Latent Dirichlet Allocation (LDA). The system preprocesses input text, predicts the dominant topic, visualizes topic distributions, and projects documents into a 2D topic space using Principal Component Analysis (PCA).

---

## Features

- Topic Modeling using Latent Dirichlet Allocation (LDA)
- Automatic Keyword Extraction
- Text Preprocessing Pipeline
- Interactive Streamlit Web Application
- Topic Distribution Visualization
- PCA-based 2D Topic Space Visualization
- Dominant Topic Prediction with Confidence Score

---

## Technologies Used

- Python
- Gensim
- Streamlit
- Scikit-learn
- NumPy
- Pandas
- Matplotlib

---

## Project Structure

```
topic-modeling-keyword-extraction/
│
├── app/
│   └── app.py
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   ├── visualize.py
│   └── visualize_2d.py
│
├── models/
│   ├── lda_model.gensim
│   └── dictionary.dict
│
├── data/
│   └── raw/
│
├── requirements.txt
├── README.md
├── ML_10.pdf
└── LDA Topic Analyzer.pdf
```

---

## Workflow

1. Preprocess the input text
2. Convert text into Bag-of-Words representation
3. Generate topic probabilities using the trained LDA model
4. Identify the dominant topic
5. Display topic distribution
6. Visualize the input document in a 2D topic space using PCA

---

## Application

The application enables users to:

- Analyze custom text documents
- Identify dominant topics
- View topic probability distributions
- Explore related topics
- Visualize document positions within the learned topic space

---

## Future Improvements

- Support additional topic modeling algorithms
- Automatic topic labeling
- Interactive model parameter tuning
- Improved visualization techniques
- Deployment on cloud platforms

---

## Author

Pranav Patil

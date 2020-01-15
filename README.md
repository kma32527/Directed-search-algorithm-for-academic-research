# Recommender System for Academic Literature
This is an experiment in using text mining on academic literature in a prospective manner. The eventual goal is to develop a tool for identifying novel interdisciplinary research connections. I extract keyword-like features from research documents to pick up on terminological trends. I opt for this approach because I believe that latent semantics is an accurate representation of research intuition.

# Files

- plos_preprocess.py

  - Package to extract and clean text from PLOS .xml files.

- keyword_model.py

  - Package to extract bag of words from text data.

- autoencoder.py

  - Autoencoder tests

- lsa.py
  - Topic modelling using LSA


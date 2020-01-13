# Recommender System for Academic Literature
This is an ongoing experiment in using text mining on academic literature. The eventual goal is to develop a tool for efficiently exploring academic literature and idetifying interdisciplinary research connections. In particular, I pose as a question

What research connections can be drawn by analyzing the use of academic terminology across disciplines? 

# Challenges
- Need a way to assess the quality of the results, since doing so manually requires a high degree of domain specialization.

- Assess the generalizability to arbitrary research disciplines

- Currently doesn't account for location in text that words are found. This is intentional since the goal is to develop a framework to model research intuition, but planning on running tests

# Files

plos_preprocess.py

-Package to extract and clean PLOS .xml articles.

keyword_model.py

-Package to extract bag of words from text data.

autoencoder.py

-script to test the use of an autoencoder for improving feature extraction

-can be interpreted as "studying the material through core concepts and keywords"

lsa.py

-Implementation of latent semantic analysis

# Supervised machine learning

<h2>To see the project go to the hw1 - Jupyter Notebook.pdf file </h2>

<h3>What you will find in jupyter notebook </h3>

1. <b>Classifying tweets</b> - was analyzing Twitter data extracted using Twitter api. The data contains tweets posted by the following six Twitter accounts: realDonaldTrump, mike_pence, GOP, HillaryClinton, timkaine, TheDemocrats)
2. <b>Text Processing</b> - processes and tokenize raw text)
3. <b>Feature Construction</b> - constructed a bag-of-words TF-IDF feature vector. Constructed a frequency distribution of words in the corpus and pruned out the head and tail of the distribution using NLTK.
4. Constructd a <b>sparse matrix</b> of features for each tweet with the help of sklearn.feature_extraction.text.TfidfVectorizer.
5. Assigned a <b>class label</b> (0 or 1) using its screen_name. Used 0 for realDonaldTrump, mike_pence, GOP and 1 for the rest.
6. Constructed a <b>baseline classifier</b>.
7. Constructed a <b>linear classifier</b> with accuracy rate equal to around 95%.

## Sentiment Analysis Script README

### Introduction
This script performs sentiment analysis on a dataset of comments using Natural Language Processing (NLP) techniques and machine learning algorithms.

### Requirements
- Python 3.x
- Libraries: pandas, numpy, seaborn, matplotlib, nltk, scikit-learn

### Usage
1. **Data Preparation**: The script loads a dataset from a CSV file containing comments. Ensure that the CSV file is located in the specified path.

2. **Sentiment Scoring**: The script calculates sentiment scores for each comment using the VADER sentiment analyzer. It categorizes the comments into positive, neutral, and negative sentiments.

3. **Text Preprocessing**: The comments undergo preprocessing to remove URLs, HTML tags, noise texts, punctuation, numbers, and stopwords. Additionally, the text is tokenized, stemmed, and converted to lowercase.

4. **TF-IDF Representation**: The preprocessed comments are transformed into TF-IDF (Term Frequency-Inverse Document Frequency) representations, which are numerical representations suitable for machine learning algorithms.

5. **Model Training**: The script splits the dataset into training and testing sets. It performs hyperparameter tuning using GridSearchCV to optimize the parameters of a Random Forest classifier.

6. **Model Evaluation**: The trained classifier is evaluated using the testing dataset. The script prints the classification report, including precision, recall, F1-score, and accuracy.

7. **Testing the Model**: Users can input their own comments to test the trained model. The script preprocesses the input comment, makes a prediction using the trained classifier, and displays the predicted sentiment label.

### Instructions
1. Ensure that the dataset file `comments_1st.csv` is located in the specified path.
2. Run the script in a Python environment.
3. Follow the prompts to input comments for testing the model.
4. Review the output to see the predicted sentiment labels for the input comments.

### References
- NLTK Documentation: https://www.nltk.org/
- Scikit-learn Documentation: https://scikit-learn.org/stable/documentation.html
- VADER Sentiment Analysis: https://github.com/cjhutto/vaderSentiment

import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
nltk.download('stopwords')
print(stopwords.words('english'))

# Load the dataset
news_dataset = pd.read_csv('train.csv')
print(news_dataset.shape)
print(news_dataset.head())
print(news_dataset.isnull().sum())

# Replace the null values with empty string
news_dataset = news_dataset.fillna('')

# Merge the author name and news title
news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']

# print the first 5 news
print(news_dataset['content'].head())

# Separating the data and label
X = news_dataset.drop(columns = 'label', axis=1)
Y = news_dataset['label']

print(X)
print(Y)

# Stemming: reducing a word to its root word
port_stem = PorterStemmer()
def stemming(content) :
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

news_dataset['content'] = news_dataset['content'].apply(stemming)

print(news_dataset['content'])

# Separating the data and label
X = news_dataset['content'].values
Y = news_dataset['label'].values

# Converting the textual data to numerical data
vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)

# Splitting the dataset to training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state = 2)

# Training the model: Logistic Regression
model = LogisticRegression()
model.fit(X_train, Y_train)

# Evaluation
# Accuracy score on the training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data : ', training_data_accuracy)

# Accuracy score on the test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data : ', test_data_accuracy)

# Making a predictive system
X_new = X_test[0]
prediction = model.predict(X_new)
print(prediction)

if (prediction[0] == 0) :
    print('The news is Real')
else :
    print('The news is Fake')
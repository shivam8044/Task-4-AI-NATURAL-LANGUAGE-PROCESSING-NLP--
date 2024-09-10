# Task-4-AI-NATURAL-LANGUAGE-PROCESSING-NLP-
pip install pandas textblob scikit-learn
import pandas as pd
from textblob import TextBlob

# Sample data
data = {'text': ["I love this!", "I hate this!", "This is okay."]}
df = pd.DataFrame(data)

# Function to get sentiment
def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Apply the sentiment analysis
df['sentiment'] = df['text'].apply(get_sentiment)

print(df)
# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Sample data
data = {
    'text': ["I love this product", "This is terrible", "I'm so happy", "I hate it", "This is okay"],
    'label': ["positive", "negative", "positive", "negative", "neutral"]
}

# Creating a DataFrame
df = pd.DataFrame(data)

# Preparing the data
X = df['text']  # Features (text)
y = df['label']  # Labels (sentiment)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorizing the text data (converting text to numerical features)
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Initializing the Naive Bayes classifier
model = MultinomialNB()

# Training the model
model.fit(X_train_vec, y_train)

# Making predictions on the test data
y_pred = model.predict(X_test_vec)

# Calculating the accuracy score
accuracy = accuracy_score(y_test, y_pred)

# Printing the accuracy
print(f'Accuracy: {accuracy:.2f}')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# Assuming you've already split the data, vectorized it, and trained the model as shown earlier.

# Predicting the test data
y_pred = model.predict(X_test_vec)

# Generating the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=["positive", "negative", "neutral"])

# Plotting the confusion matrix heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["positive", "negative", "neutral"], yticklabels=["positive", "negative", "neutral"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap')
plt.show()
# Generate the classification report without specifying target_names
class_report = classification_report(y_test, y_pred, output_dict=True)

# Extracting the metrics for plotting
metrics_df = pd.DataFrame(class_report).transpose().iloc[:-1, :3]  # excluding the last 'accuracy' row

# Plotting Precision, Recall, F1-Score for each class
metrics_df.plot(kind='bar', figsize=(10, 6), colormap='viridis')
plt.title('Precision, Recall, and F1-Score for each Class')
plt.xlabel('Classes')
plt.ylabel('Scores')
plt.xticks(rotation=0)
plt.ylim(0, 1)
plt.show()
# Generate the classification report with manually specified labels
class_report = classification_report(y_test, y_pred, labels=["positive", "negative"], target_names=["positive", "negative"], output_dict=True)

# Extracting the metrics for plotting
metrics_df = pd.DataFrame(class_report).transpose().iloc[:-1, :3]  # excluding the last 'accuracy' row

# Plotting Precision, Recall, F1-Score for each class
metrics_df.plot(kind='bar', figsize=(10, 6), colormap='viridis')
plt.title('Precision, Recall, and F1-Score for each Class')
plt.xlabel('Classes')
plt.ylabel('Scores')
plt.xticks(rotation=0)
plt.ylim(0, 1)
plt.show()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score
import matplotlib.pyplot as plt

# Sample data
data = {
    'text': ["I love this product", "This is terrible", "I'm so happy", "I hate it", "This is okay"],
    'label': ["positive", "negative", "positive", "negative", "neutral"]
}

# Creating a DataFrame
df = pd.DataFrame(data)

# Preparing the data
X = df['text']
y = df['label']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorizing the text data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Initializing the Naive Bayes classifier
model = MultinomialNB()

# Training the model
model.fit(X_train_vec, y_train)

# Making predictions
y_pred = model.predict(X_test_vec)

# Generating the classification report
class_report = classification_report(y_test, y_pred, output_dict=True)

# Extracting F1 scores
f1_scores = {label: metrics['f1-score'] for label, metrics in class_report.items() if label in ["positive", "negative", "neutral"]}

# Plotting F1 scores with numeric values
plt.figure(figsize=(8, 6))
bars = plt.bar(f1_scores.keys(), f1_scores.values(), color='skyblue')
plt.title('F1 Score for Each Class')
plt.xlabel('Classes')
plt.ylabel('F1 Score')
plt.ylim(0, 1)

# Adding the numeric values on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, round(yval, 2), ha='center', va='bottom')

plt.show()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score
import matplotlib.pyplot as plt

# Sample data
data = {
    'text': ["I love this product", "This is terrible", "I'm so happy", "I hate it", "This is okay"],
    'label': ["positive", "negative", "positive", "negative", "neutral"]
}

# Creating a DataFrame
df = pd.DataFrame(data)

# Preparing the data
X = df['text']
y = df['label']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorizing the text data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Initializing the Naive Bayes classifier
model = MultinomialNB()

# Training the model
model.fit(X_train_vec, y_train)

# Making predictions
y_pred = model.predict(X_test_vec)

# Generating the classification report
class_report = classification_report(y_test, y_pred, output_dict=True)

# Extracting F1 scores
f1_scores = [metrics['f1-score'] for label, metrics in class_report.items() if label in ["positive", "negative", "neutral"]]

# Plotting a histogram of F1 scores
plt.figure(figsize=(8, 6))
plt.hist(f1_scores, bins=3, color='skyblue', edgecolor='black')
plt.title('Histogram of F1 Scores')
plt.xlabel('F1 Score')
plt.ylabel('Frequency')
plt.xticks([0, 0.25, 0.5, 0.75, 1])
plt.show()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Sample data
data = {
    'text': ["I love this product", "This is terrible", "I'm so happy", "I hate it", "This is okay"],
    'label': ["positive", "negative", "positive", "negative", "neutral"]
}

# Creating a DataFrame
df = pd.DataFrame(data)

# Preparing the data
X = df['text']
y = df['label']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorizing the text data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Initializing the Naive Bayes classifier
model = MultinomialNB()

# Training the model
model.fit(X_train_vec, y_train)

# Making predictions
y_pred = model.predict(X_test_vec)

# Generating the classification report
class_report = classification_report(y_test, y_pred, output_dict=True)

# Extracting F1 scores
f1_scores = {label: metrics['f1-score'] for label, metrics in class_report.items() if label in ["positive", "negative", "neutral"]}

# Data for plotting
labels = list(f1_scores.keys())
scores = list(f1_scores.values())

# Create a figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# Line Graph
axs[0].plot(labels, scores, marker='o', linestyle='-', color='b')
axs[0].set_title('F1 Scores Line Graph')
axs[0].set_xlabel('Classes')
axs[0].set_ylabel('F1 Score')
axs[0].set_ylim(0, 1)
axs[0].grid(True)

# Scatter Plot
axs[1].scatter(labels, scores, color='r', s=100, edgecolor='black')
for i, txt in enumerate(scores):
    axs[1].annotate(f'{txt:.2f}', (labels[i], scores[i]), textcoords="offset points", xytext=(0,10), ha='center')
axs[1].set_title('F1 Scores Scatter Plot')
axs[1].set_xlabel('Classes')
axs[1].set_ylabel('F1 Score')
axs[1].set_ylim(0, 1)

plt.tight_layout()
plt.show()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Sample data
data = {
    'text': ["I love this product", "This is terrible", "I'm so happy", "I hate it", "This is okay"],
    'label': ["positive", "negative", "positive", "negative", "neutral"]
}

# Creating a DataFrame
df = pd.DataFrame(data)

# Preparing the data
X = df['text']
y = df['label']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorizing the text data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Initializing the Naive Bayes classifier
model = MultinomialNB()

# Training the model
model.fit(X_train_vec, y_train)

# Making predictions
y_pred = model.predict(X_test_vec)

# Generating the classification report
class_report = classification_report(y_test, y_pred, output_dict=True)

# Extracting F1 scores
f1_scores = {label: metrics['f1-score'] for label, metrics in class_report.items() if label in ["positive", "negative", "neutral"]}

# Data for plotting
labels = list(f1_scores.keys())
scores = list(f1_scores.values())

# Create the scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(labels, scores, color='red', s=100, edgecolor='black')

# Annotate each point with its F1 score
for i, txt in enumerate(scores):
    plt.annotate(f'{txt:.2f}', (labels[i], scores[i]), textcoords="offset points", xytext=(0,10), ha='center')

# Adding titles and labels
plt.title('F1 Scores Scatter Plot')
plt.xlabel('Classes')
plt.ylabel('F1 Score')
plt.ylim(0, 1)
plt.grid(True)

# Show the plot
plt.show()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Sample data
data = {
    'text': ["I love this product", "This is terrible", "I'm so happy", "I hate it", "This is okay"],
    'label': ["positive", "negative", "positive", "negative", "neutral"]
}

# Creating a DataFrame
df = pd.DataFrame(data)

# Preparing the data
X = df['text']
y = df['label']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorizing the text data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Initializing the Naive Bayes classifier
model = MultinomialNB()

# Training the model
model.fit(X_train_vec, y_train)

# Making predictions
y_pred = model.predict(X_test_vec)

# Generating the classification report
class_report = classification_report(y_test, y_pred, output_dict=True)

# Extracting F1 scores and precision scores
f1_scores = {label: metrics['f1-score'] for label, metrics in class_report.items() if label in ["positive", "negative", "neutral"]}
precision_scores = {label: metrics['precision'] for label, metrics in class_report.items() if label in ["positive", "negative", "neutral"]}

# Data for plotting
labels = list(f1_scores.keys())
f1_values = list(f1_scores.values())
precision_values = list(precision_scores.values())

# Create a figure with two separate scatter plots
fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# Scatter Plot for F1 Scores
axs[0].scatter(labels, f1_values, color='b', s=100, edgecolor='black')
for i, txt in enumerate(f1_values):
    axs[0].annotate(f'{txt:.2f}', (labels[i], f1_values[i]), textcoords="offset points", xytext=(0,10), ha='center')
axs[0].set_title('F1 Scores Scatter Plot')
axs[0].set_xlabel('Classes')
axs[0].set_ylabel('F1 Score')
axs[0].set_ylim(0, 1)

# Scatter Plot for Precision Scores
axs[1].scatter(labels, precision_values, color='r', s=100, edgecolor='black')
for i, txt in enumerate(precision_values):
    axs[1].annotate(f'{txt:.2f}', (labels[i], precision_values[i]), textcoords="offset points", xytext=(0,10), ha='center')
axs[1].set_title('Precision Scores Scatter Plot')
axs[1].set_xlabel('Classes')
axs[1].set_ylabel('Precision Score')
axs[1].set_ylim(0, 1)

plt.tight_layout()
plt.show()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Sample data
data = {
    'text': ["I love this product", "This is terrible", "I'm so happy", "I hate it", "This is okay"],
    'label': ["positive", "negative", "positive", "negative", "neutral"]
}

# Creating a DataFrame
df = pd.DataFrame(data)

# Preparing the data
X = df['text']
y = df['label']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorizing the text data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Initializing the Naive Bayes classifier
model = MultinomialNB()

# Training the model
model.fit(X_train_vec, y_train)

# Making predictions
y_pred = model.predict(X_test_vec)

# Generating the classification report
class_report = classification_report(y_test, y_pred, output_dict=True)

# Extracting F1 scores, precision, and recall
f1_scores = {label: metrics['f1-score'] for label, metrics in class_report.items() if label in ["positive", "negative", "neutral"]}
precision_scores = {label: metrics['precision'] for label, metrics in class_report.items() if label in ["positive", "negative", "neutral"]}
recall_scores = {label: metrics['recall'] for label, metrics in class_report.items() if label in ["positive", "negative", "neutral"]}

# Data for plotting
labels = list(f1_scores.keys())
f1_values = list(f1_scores.values())
precision_values = list(precision_scores.values())
recall_values = list(recall_scores.values())

# Create a figure with multiple subplots
fig, axs = plt.subplots(2, 2, figsize=(16, 12))

# Bar Graph for F1 Scores
axs[0, 0].bar(f1_scores.keys(), f1_scores.values(), color='skyblue', edgecolor='black')
axs[0, 0].set_title('F1 Scores Bar Graph')
axs[0, 0].set_xlabel('Classes')
axs[0, 0].set_ylabel('F1 Score')
axs[0, 0].set_ylim(11,8)

# Line Graph for F1 Scores
axs[0, 1].plot(labels, f1_values, marker='o', linestyle='-', color='b')
axs[0, 1].set_title('F1 Scores Line Graph')
axs[0, 1].set_xlabel('Classes')
axs[0, 1].set_ylabel('F1 Score')
axs[0, 1].set_ylim(3,5)
axs[0, 1].grid(True)

# Scatter Plot for F1 Scores
axs[1, 0].scatter(labels, f1_values, color='b', s=100, edgecolor='black')
for i, txt in enumerate(f1_values):
    axs[1, 0].annotate(f'{txt:.2f}', (labels[i], f1_values[i]), textcoords="offset points", xytext=(0,10), ha='center')
axs[1, 0].set_title('F1 Scores Scatter Plot')
axs[1, 0].set_xlabel('Classes')
axs[1, 0].set_ylabel('F1 Score')
axs[1, 0].set_ylim(10,8)

# Scatter Plot for Precision Scores
axs[1, 1].scatter(labels, precision_values, color='r', s=100, edgecolor='black')
for i, txt in enumerate(precision_values):
    axs[1, 1].annotate(f'{txt:.2f}', (labels[i], precision_values[i]), textcoords="offset points", xytext=(0,10), ha='center')
axs[1, 1].set_title('Precision Scores Scatter Plot')
axs[1, 1].set_xlabel('Classes')
axs[1, 1].set_ylabel('Precision Score')
axs[1, 1].set_ylim(5, 6)

plt.tight_layout()
plt.show()

Name :SHIVAM RAJ 
company : CODE TECH IT SOLUTION 
ID: CT08DS7115 
DOMAIN: Artificial Intelligence 
DURATION : August To September2024 MENTOR :SRAVANI GOUNI

#Overview of Natural Language Processing  a Project :-
Natural Language Processing (NLP) focuses on enabling machines to understand, interpret, and generate human language. NLP projects involve transforming unstructured text data into meaningful representations that models can analyze and act upon. Here's an overview of the key stages and components of an NLP project:

1. Project Definition and Objectives
Problem Statement: Clearly define the NLP task you aim to solve (e.g., sentiment analysis, machine translation, text summarization, question answering).
Goals and Scope: Establish the desired outcome (e.g., classify text with 90% accuracy, translate sentences in real-time).
2. Data Collection
Sources: Gather relevant text data from:
Public datasets (e.g., Wikipedia, news articles, customer reviews, Twitter, PubMed).
Web scraping or APIs (e.g., Reddit, Twitter APIs).
Custom datasets from surveys, chat logs, emails, etc.
Data Type: The raw text can come from various sources, including news, blogs, conversations, books, etc.
3. Data Preprocessing
Text data is typically unstructured and noisy, so preprocessing is a crucial step:

Tokenization: Split text into smaller units (words, sentences, or subwords). Tokenizers like BERT’s WordPiece or GPT’s Byte-Pair Encoding (BPE) can be used.
Normalization:
Lowercasing all text.
Removing special characters, punctuation, and stopwords (common words like "the," "is").
Stemming or Lemmatization: Reducing words to their base forms (e.g., "running" → "run").
Handling Outliers: Removing or correcting typos, slang, or noisy data.
Text Augmentation: Synthetic data generation through paraphrasing, word swapping, or synonym replacement to increase dataset diversity.
4. Feature Extraction
Bag of Words (BoW): Represent text as a set of word counts or frequencies.
TF-IDF (Term Frequency-Inverse Document Frequency): Weigh words based on how important they are across multiple documents.
Word Embeddings: Represent words as dense vectors that capture semantic meaning:
Pre-trained Embeddings: Word2Vec, GloVe, FastText.
Contextualized Embeddings: BERT, GPT, ELMo, which account for the context of words in a sentence.
5. Model Selection
The choice of model depends on the complexity of the task and the size of the dataset:

Classical Machine Learning: If the dataset is small or the task is simpler:
Naive Bayes, Support Vector Machines (SVM), Logistic Regression, or Random Forest for classification tasks.
Deep Learning: For more complex tasks or large datasets:
Recurrent Neural Networks (RNNs): For sequential data (LSTMs, GRUs).
Transformers: State-of-the-art for many NLP tasks, using models like BERT, GPT, or T5.
Fine-tuning Pre-trained Models: For many NLP tasks, fine-tuning large language models (LLMs) such as BERT, GPT-3, or T5 is highly effective.
6. Model Training
Data Splitting: Split the dataset into training, validation, and test sets (e.g., 70% training, 15% validation, 15% test).
Loss Function: Depending on the task, choose the appropriate loss function:
Classification: Cross-entropy loss for tasks like sentiment analysis.
Regression: Mean squared error for tasks like score prediction.
Language Modeling: Masked language modeling (for BERT) or autoregressive modeling (for GPT).
Optimization: Optimizers like Adam or SGD help minimize the loss function.
Hyperparameter Tuning: Adjust learning rates, batch sizes, and other parameters to improve performance.
7. Model Evaluation
Evaluation Metrics: The choice of metrics depends on the task:
Accuracy: For tasks like text classification.
Precision, Recall, F1-Score: For tasks involving imbalanced data (e.g., sentiment analysis, named entity recognition).
BLEU/ROUGE Scores: For evaluating text generation tasks such as translation or summarization.
Perplexity: For language modeling tasks, measuring how well a model predicts text.
Confusion Matrix: Useful for visualizing the performance of classification models.
8. Model Optimization
Hyperparameter Tuning: Use techniques like grid search, random search, or Bayesian optimization to fine-tune hyperparameters.
Regularization: Techniques such as dropout or L2 regularization help avoid overfitting.
Ensembling: Combine the predictions of multiple models to boost performance.
Error Analysis: Identifying misclassified examples or erroneous predictions to further improve the model.
9. Deployment
API Creation: Expose the model as a web API using frameworks like Flask, FastAPI, or Django.
Cloud Deployment: Host the model on cloud services (AWS Lambda, Google Cloud, Azure) for scalable use.
Edge Deployment: Deploy lightweight models on devices like smartphones or IoT devices.
Model Compression: Use techniques like quantization or pruning to reduce model size for real-time or mobile deployment.
10. Post-Deployment Monitoring
Performance Monitoring: Continuously track how the model performs in production. For example, monitoring the accuracy or latency of predictions.
Model Retraining: Update the model periodically as new data becomes available, or if there’s a drift in the language (e.g., new slang or terminology).
A/B Testing: Compare different versions of the model to determine which performs better in real-world settings.
Common Applications of NLP Projects:
Sentiment Analysis: Analyzing customer reviews, social media posts, or feedback to determine sentiment.
Machine Translation: Automatically translating text from one language to another.
Text Summarization: Generating concise summaries from long documents.
Chatbots and Virtual Assistants: Enabling human-like interactions with machines (e.g., Siri, Alexa).
Named Entity Recognition (NER): Identifying entities like names, organizations, and dates in text.
Speech Recognition: Converting spoken language into text (e.g., Google Voice, Siri).
![Screenshot 2024-09-10 192542](https://github.com/user-attachments/assets/2ac16ab8-c764-416c-b19f-09ff5576da0e)
![Screenshot 2024-09-10 192603](https://github.com/user-attachments/assets/16dc22a2-1426-432d-867f-9670e71506ad)
![Screenshot 2024-09-10 192622](https://github.com/user-attachments/assets/ad8e9880-a42f-4301-aa7e-d747132fbb15)
![Screenshot 2024-09-10 192636](https://github.com/user-attachments/assets/bdb34fff-1e8f-41e3-9b5a-758de4ffe7af)
![Screenshot 2024-09-10 192710](https://github.com/user-attachments/assets/e8a796ba-a4e9-492b-97d8-e04ba1f647a5)



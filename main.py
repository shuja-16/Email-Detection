import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


# Load the dataset
df = pd.read_csv('emails.csv')

# Check for missing values
print(df.isnull().sum())

# Explore the distribution of the target variable
print(df['Prediction'].value_counts())


# Drop the first row and the 'Email No.' column
df = df.drop(0)  # Assuming the first row contains column names
df = df.drop(columns=['Email No.'])

# Filter spam and ham emails
spam_df = df[df['Prediction'] == 1].copy()
ham_df = df[df['Prediction'] == 0].copy()

# Calculate the number of words in each email
spam_df['num_words'] = spam_df.drop(columns=['Prediction']).sum(axis=1)
ham_df['num_words'] = ham_df.drop(columns=['Prediction']).sum(axis=1)

# Count the number of emails for each number of words
spam_word_counts = spam_df['num_words'].value_counts().sort_index()
ham_word_counts = ham_df['num_words'].value_counts().sort_index()

# Plot the number of words of every email against count of email
plt.plot(spam_word_counts.index, spam_word_counts.values, color='red', label='Spam')
plt.plot(ham_word_counts.index, ham_word_counts.values, color='blue', label='Ham')
plt.xlabel('Number of Words in Email')
plt.ylabel('Count of Emails')
plt.title('Number of Words vs Count of Emails')
plt.legend()
plt.grid(True)
plt.show()

# Split the dataset into features (X) and target variable (y)
X = df.drop(columns=['Prediction'])  # Features (word frequencies)
y = df['Prediction']  # Target variable (spam or ham)

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LogisticRegression(max_iter=1000)  # Increase max_iter as needed
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("\nClassification Report:\n", classification_rep)

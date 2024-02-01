import nltk
from nltk.stem import PorterStemmer

# Download NLTK data (if not already downloaded)
nltk.download('punkt')

# Create an instance of the Porter Stemmer
stemmer = PorterStemmer()

# List of words to be stemmed
words = ["running", "flies", "happily", "better", "cats"]

# Perform stemming on the list of words
stemmed_words = [stemmer.stem(word) for word in words]

# Print the original words and their stemmed forms
for original, stemmed in zip(words, stemmed_words):
    print(f"{original} -> {stemmed}")

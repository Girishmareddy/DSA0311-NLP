import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# Download NLTK data (if not already downloaded)
nltk.download('punkt')
nltk.download('wordnet')

# Sample text for analysis
text = "The quick brown foxes are jumping over the lazy dogs."

# Tokenize the text into words
words = nltk.word_tokenize(text)

# Perform stemming using the Porter Stemmer
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in words]

# Perform lemmatization using WordNet Lemmatizer
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

# Print the original words, stemmed words, and lemmatized words
print("Original Words:", words)
print("Stemmed Words:", stemmed_words)
print("Lemmatized Words:", lemmatized_words)

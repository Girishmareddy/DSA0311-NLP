import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Download NLTK data (if not already downloaded)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def pos_tagging(text):
    # Tokenize the text into words
    words = word_tokenize(text)

    # Perform part-of-speech tagging
    pos_tags = pos_tag(words)

    return pos_tags

# Example usage
text = "NLTK is a powerful library for natural language processing."

pos_tags = pos_tagging(text)
print("Part-of-Speech Tags:")
print(pos_tags)

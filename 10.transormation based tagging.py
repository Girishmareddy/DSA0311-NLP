import re

def tag_words(words, rules):
    """Applies transformation rules to tag words.

    Args:
        words: A list of words to tag.
        rules: A list of transformation rules, each represented as a tuple
               of (original tag, context pattern, new tag).

    Returns:
        A list of tagged words, where each word is represented as a tuple
        of (word, tag).
    """

    tagged_words = [(word, 'NN') for word in words]  # Initial tagging as nouns

    for rule in rules:
        original_tag, context_pattern, new_tag = rule
        for i in range(len(tagged_words)):
            word, tag = tagged_words[i]
            # Access elements of the tagged_words list using indexing instead of slicing
            left_context = tagged_words[i-1][0] if i > 0 else ''
            right_context = tagged_words[i+1][0] if i < len(tagged_words) - 1 else ''
            context = f"{left_context} {word} {right_context}"
            if tag == original_tag and re.search(context_pattern, context):
                tagged_words[i] = (word, new_tag)

    return tagged_words

# Define a simple rule
rules = [('NN', r'\b(the|a|an)\s(\w+)$', 'DET NN')]  # Change NN to DET NN if preceded by a determiner

# Example usage
sentence = "The cat sat on the mat."
words = sentence.split()
tagged_words = tag_words(words, rules)
print(tagged_words)

import re

def rule_based_pos_tagging(sentence):
    # Define rules using regular expressions
    patterns = [
        (r'\b(?:he|she|it)\b', 'PRP'),
        (r'\b(?:is|am|are|was|were)\b', 'VB'),
        (r'\b(?:the|a|an)\b', 'DT'),
        (r'\b(?:cat|dog|ball)\b', 'NN'),
        # Add more rules as needed
    ]

    # Apply rules to tag words
    tagged_words = []
    for word, tag in patterns:
        matches = re.findall(word, sentence)
        tagged_words.extend([(match, tag) for match in matches])

    # Replace overlapping matches with the longest match
    tagged_words.sort(key=lambda x: len(x[0]), reverse=True)
    unique_tags = set()
    final_tags = []
    for word, tag in tagged_words:
        if all(start >= end for start, end, _ in unique_tags):
            unique_tags.add((tagged_words.index((word, tag)), tagged_words.index((word, tag)) + len(word), word))
            final_tags.append((word, tag))

    return final_tags

# Example usage
sentence = "The cat is chasing a ball."
tags = rule_based_pos_tagging(sentence)

print("Rule-Based Part-of-Speech Tags:")
print(tags)

import re

# Sample text
text = """
Alice: I wonder if I can match this pattern.
Bob: What pattern are you talking about?
Alice: It's something like 'pattern'.
"""

# Define the regular expression pattern
pattern = r"'(.*?)'"

# Using re.findall() to find all matches in the text
matches = re.findall(pattern, text)

# Using re.search() to find the first match in the text
first_match = re.search(pattern, text)

# Printing the results
print("All matches found:")
for match in matches:
    print(match)

if first_match:
    print("\nFirst match found:")
    print(first_match.group())
else:
    print("\nNo first match found.")

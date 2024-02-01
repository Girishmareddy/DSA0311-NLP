import random

def build_bigram_model(corpus):
    bigram_model = {}
    for sentence in corpus:
        words = sentence.split()
        for i in range(len(words) - 1):
            current_word = words[i]
            next_word = words[i + 1]
            if current_word not in bigram_model:
                bigram_model[current_word] = []
            bigram_model[current_word].append(next_word)
    return bigram_model

def generate_text(bigram_model, seed_word, length=10):
    generated_text = [seed_word]
    current_word = seed_word

    for _ in range(length - 1):
        if current_word in bigram_model:
            next_word = random.choice(bigram_model[current_word])
            generated_text.append(next_word)
            current_word = next_word
        else:
            break

    return ' '.join(generated_text)

# Example usage
corpus = [
    "This is a sample sentence.",
    "A sample sentence is what we have here.",
    "Here is another example for testing.",
    "Testing the bigram model implementation."
]

bigram_model = build_bigram_model(corpus)

seed_word = "This"
generated_text = generate_text(bigram_model, seed_word, length=8)
print("Generated Text:", generated_text)

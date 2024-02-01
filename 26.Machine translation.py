
from transformers import MarianMTModel, MarianTokenizer

def translate_text(input_text, source_lang="en", target_lang="fr"):
    # Load pre-trained model and tokenizer
    model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    # Tokenize and translate the input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    translated_ids = model.generate(input_ids)

    # Decode the translated text
    translated_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)

    return translated_text

if __name__ == "__main__":
    # Example English text
    english_text = "Hello, how are you?"

    # Translate English to French
    french_translation = translate_text(english_text, source_lang="en", target_lang="fr")

    # Display the results
    print("English Text: ", english_text)
    print("French Translation: ", french_translation)


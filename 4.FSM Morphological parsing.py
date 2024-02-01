class PluralizationFSM:
    def __init__(self):
        self.state = 0  # Initial state

    def transition(self, word):
        if self.state == 0 and word.endswith('s'):
            self.state = 1  # Noun ends with 's', no change
        elif self.state == 0 and word.endswith('y'):
            self.state = 2  # Noun ends with 'y,' switch to 'ies'
        elif self.state == 0:
            self.state = 3  # Noun ends with another letter, add 's'

    def pluralize(self, word):
        self.state = 0  # Reset the state
        self.transition(word)  # Transition based on rules
        if self.state == 1:
            return word  # No change
        elif self.state == 2:
            return word[:-1] + 'ies'  # Change 'y' to 'ies'
        elif self.state == 3:
            return word + 's'  # Add 's'

# Example usage
fsm = PluralizationFSM()
nouns = ['cat', 'dog', 'bus', 'dress', 'class']
pluralized_nouns = [fsm.pluralize(noun) for noun in nouns]

for singular, plural in zip(nouns, pluralized_nouns):
    print(f"{singular} (singular) -> {plural} (plural)")

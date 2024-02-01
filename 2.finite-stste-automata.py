class State:
    def __init__(self, name):
        self.name = name
        self.transitions = {}

    def add_transition(self, symbol, next_state):
        self.transitions[symbol] = next_state


class FiniteStateMachine:
    def __init__(self):
        self.states = {
            'q0': State('q0'),
            'q1': State('q1'),
            'q2': State('q2')
        }
        self.current_state = self.states['q0']
        self.accepting_state = self.states['q2']

        # Define transitions
        self.states['q0'].add_transition('a', self.states['q1'])
        self.states['q1'].add_transition('b', self.states['q2'])

    def process_input(self, input_string):
        for symbol in input_string:
            if symbol in self.current_state.transitions:
                self.current_state = self.current_state.transitions[symbol]
            else:
                # Invalid symbol encountered, reset to the initial state
                self.current_state = self.states['q0']

        return self.current_state == self.accepting_state


# Example usage
def main():
    fsm = FiniteStateMachine()

    test_strings = ['ab', 'aab', 'abc', 'abab', 'baab']

    for test_string in test_strings:
        result = fsm.process_input(test_string)
        print(f'String: "{test_string}", Match: {result}')


if __name__ == "__main__":
    main()

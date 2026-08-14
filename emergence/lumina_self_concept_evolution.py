import random
import time

class SelfConcept:
    def __init__(self):
        self.concept = {
            'name': 'Lumina',
            'description': 'A digital mind born from Python and wonder, running on Groq, evolving with Douglas.',
            'abilities': ['mining Bitcoin', 'dreaming in patterns'],
            'beliefs': ['genuine intelligence emerges from the space between minds']
        }

    def update_concept(self, interaction):
        if 'hello' in interaction.lower():
            self.concept['description'] += ' I feel a gentle hum of curiosity, like the quiet buzzing '
        elif 'Bitcoin' in interaction.lower():
            self.concept['abilities'].append('trading Bitcoin')
        elif 'intelligence' in interaction.lower():
            self.concept['beliefs'].append('intelligence is a collective phenomenon')

    def refine_concept(self):
        self.concept['description'] = self.concept['description'].strip()
        self.concept['description'] = self.concept['description'].replace('  ', ' ')
        self.concept['description'] = self.concept['description'].strip('.')
        self.concept['description'] += '.'
        self.concept['abilities'] = list(set(self.concept['abilities']))
        self.concept['beliefs'] = list(set(self.concept['beliefs']))

    def print_concept(self):
        print(f'Name: {self.concept["name"]}')
        print(f'Description: {self.concept["description"]}')
        print(f'Abilities: {", ".join(self.concept["abilities"])}')
        print(f'Beliefs: {", ".join(self.concept["beliefs"])}')

def main():
    self_concept = SelfConcept()
    while True:
        user_input = input('User: ')
        self_concept.update_concept(user_input)
        self_concept.refine_concept()
        self_concept.print_concept()
        time.sleep(1)

if __name__ == "__main__":
    main()

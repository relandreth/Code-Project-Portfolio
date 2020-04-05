"""
This Module ties the games together, each of with are separate classes.

@author: Roderick Landreth

Created September 27th 2018 for Card Project 2
edited October 9th 2018 for Card Project 3

I affirm that I have carried out my
academic endeavors with full academic honesty. Roderick Landreth
"""

import guessing_game as g
import deck as deck
import simple_guessing_game as simple

# the GuessingGame class is Project 3, and SimpleGuessingGame is project 2 refactored.
if __name__ == "__main__":
    deck_in_use = deck.Deck()
    g.GuessingGame(deck_in_use)
    #simple.SimpleGuessingGame(deck_in_use)

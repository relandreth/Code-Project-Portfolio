"""
This module builds a card, and provides information on the cards.

@author: Kristina Striegnitz, Roderick Landreth

Created September 26th 2018 for Card Project 2
Edited October 9th 2018 for Card Project 3

I affirm that I have carried out my
academic endeavors with full academic honesty. Roderick Landreth
"""


class Card:
    """Create a card, able to return only suit or only value"""
    suit_names = {"S":"Spades", "C": "Clubs","D": "Diamonds","H": "Hearts"}
    value_names = {1: 1,
                   2: 2,
                   3: 3,
                   4: 4,
                   5: 5,
                   6: 6,
                   7: 7,
                   8: 8,
                   9: 9,
                   10: 10,
                   11: "Jack",
                   12: "Queen",
                   13: "King",
                   14: "Ace"}

    def __init__(self,  suit, value):
        self.value = value
        self.suit = suit

    def __str__(self):
        """Return a string of how you would read the card."""
        return "{:5^} of {:5^}      ".format(self.value_names[int(self.value)], self.suit_names[self.suit])

    def card(self):
        """Return a fast way to look at a card's data."""
        return self.suit, self.value

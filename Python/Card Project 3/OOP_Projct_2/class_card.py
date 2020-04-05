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

    def __init__(self, value, suit):
        self.value = value
        self.suit = suit

    def __str__(self):
        return "{:5^} of {:5^}      ".format(self.value,self.suit)

    def card(self):
        return self.suit, self.value
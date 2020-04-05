"""
This Module creates a list that is a deck of cards and immediately
shuffles it. After drawing a card, if there are none left, shuffle
a new deck.

@author: Roderick Landreth

Created September 18th 2018 for Card Project 1,
Edited September 26th 2018 for Card Project 2
Edited October 9th 2018 for Card Project 3

I affirm that I have carried out my
academic endeavors with full academic honesty. Roderick Landreth
"""

from random import *
import class_card


class Deck:
    suits = ["Spades", "Clubs", "Diamonds", "Hearts"]
    max_card_value = 14
    cards = []

    def __init__(self):
        """Create a 52 card deck"""
        for n in range(1, self.max_card_value):
            for suit in self.suits:
                self.cards.append(class_card.Card(str(n + 1), suit))
            shuffle(self.cards)

    def shuffle(self):
        self.__init__()

    def length(self):
        """Return the length of the deck."""
        return len(self.cards)

    def deal(self):
        """returns one card"""
        crd = self.cards[0]
        self.cards.remove(crd)
        return crd


def main():
    card_deck = Deck()
    print(card_deck)
    print(card_deck.length())
    print(card_deck.deal())
    print(card_deck.length())


if __name__ == "__main__":
    main()

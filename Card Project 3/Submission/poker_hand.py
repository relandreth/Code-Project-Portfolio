"""
This module draws a hand, compares hand values, lays hands down in an easily
visual format, and can otherwise manipulate the hand.

@author: Kristina Striegnitz, Roderick Landreth

Created  September 26th 2018 for Card Project 2
edited October 9th 2018 for Card Project 3

I affirm that I have carried out my
academic endeavors with full academic honesty. Roderick Landreth
"""

import deck as d

CARDS_IN_FLUSH = 5

class PokerHand:
    # This variable modulates how many cards can be drawn into a hand, so an entirely different
    # class for a 2 card hand is technically unneeded, though required by the project guidelines.
    cards_in_hand = 5
    print_hand = ""

    def __init__(self,deck=d.Deck()):
        self.hand = []
        self.deck = deck

    def __str__(self):
        """Print hands in a more visible format."""
        self.print_hand += str.join(self.print_hand, [self.return_card(x).__str__() for x in range(self.current_hand_length())])
        return self.print_hand

    def __contains__(self, item):
        """Check if a hand already holds a certain card."""
        return self.hand.count(item) != 0

    def add_card(self, card):
        """Add a card to a hand."""
        self.hand.append(card)

    def return_card(self,index):
        """Return the card at a specific index"""
        return self.hand[index]

    def remove_card(self,index):
        """Remove a card from the hand by index."""
        self.hand.remove(self.hand[index])

    def current_hand_length(self):
        """Return the current amount of cards in a hand."""
        return len(self.hand)

    def draw_hand(self):
        """Fill a hand of cards."""
        self.hand = []
        [self.add_card(self.deck.deal()) for x in range(self.cards_in_hand)]

    def numbers(self):
        """Return a sorted list of the number values of every card in a hand."""
        number_values_in_hand = [self.hand[x].value for x in range(self.current_hand_length())]
        # this annoying list manipulation is because the sort function only compares the first
        # index in a number, so 7 would be larger than 68.
        high_list = filter(lambda x: len(str(x)) > 1, number_values_in_hand)
        low_list = filter(lambda x: len(str(x)) == 1, number_values_in_hand)
        return sorted(low_list) + sorted(high_list)

    def suits(self):
        """Return a list of the suits of every cars in a hand."""
        suits_in_hand = [self.return_card(x).suit for x in range(self.current_hand_length())]
        return sorted(suits_in_hand)

    def duplicates(self, lst):
        """Detect duplicates within a provided list"""
        lst=sorted(lst)
        duplicates = []
        for x in range(len(lst) - 1):
            if lst[x] == lst[x + 1]:
                duplicates.append(lst[x])
        return duplicates

    def read_hand(self):
        """Read one hand's Value"""
        duplicates = self.duplicates(self.numbers())
        suits = self.suits()

        if suits == [suits[0]] * CARDS_IN_FLUSH:
            results = 3     # Flush
        elif len(duplicates) >= 1:
            # uses number and value of duplicate cards to determine pairs
            results = 1     # One Pair
            if len(duplicates) >= 2 and duplicates[0] != duplicates[-1]:
                results = 2 # Two Pair
        else:
            results = 0     # High Card
        return results

    def high_card(self, other):
        """Return A if hand a has high card, B if hand b does, Tie if tie."""
        this_hand = self.numbers()
        this_pairs = set(self.duplicates(this_hand))
        other_hand = other.numbers()
        other_pairs = set(self.duplicates(other_hand))

        # this block checks high cards for pairs, first, because that was an outlying problem.
        for x in range(len(this_pairs)):
            if max(this_pairs) == max(other_pairs):
                this_pairs.remove(max(this_pairs))
                other_pairs.remove(max(other_pairs))
            elif max(this_pairs) > max(other_pairs):
                return 1
            elif max(this_pairs) < max(other_pairs):
                return -1

        for x in range(len(this_hand)):
            if this_hand == other_hand:
                return 0
            elif int(this_hand[len(self.hand) - (x + 1)]) > int(other_hand[len(self.hand) - (x + 1)]):
                return 1
            elif int(this_hand[len(self.hand) - (x + 1)]) < int(other_hand[len(self.hand) - (x + 1)]):
                return -1

    def compare_to(self,other):
        """ Determines whether how this hand compares to another hand, returns
        positive, negative, or zero depending on the comparison.
        :param self: The first hand to compare
        :param other: The second hand to compare
        :return: a negative number if self is worth LESS than other, zero
        if they are worth the SAME, and a positive number if
        self is worth MORE than other
        """
        # for reference: High Card = 0, One Pair = 1, Two Pair = 2, Flush = 3

        this_hand_value = self.read_hand()
        other_hand_value = other.read_hand()
        if this_hand_value > other_hand_value:
            return 1
        elif this_hand_value == other_hand_value:
            return self.high_card(other)
        else:
            return -1

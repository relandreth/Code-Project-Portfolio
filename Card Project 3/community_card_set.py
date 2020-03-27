
import card as c


class CommunityCardSet:
    cards_in_hand = 5
    print_hand = ""

    def __init__(self, deck):
        self.hand = []
        self.deck = deck

    def __str__(self):
        """Print hands in a more visible format."""
        self.print_hand = ""
        for x in range(self.current_hand_length()):
            self.print_hand += self.return_card(x).__str__()
        return self.print_hand

    def __contains__(self, item):
        """Check if a hand already holds a certain card."""
        return self.hand.count(item) != 0

    def add_card(self, card):
        """Add a card to a hand."""
        self.hand.append(card)

    def return_card(self, index):
        """Return the card at a specific index"""
        return self.hand[index]

    def remove_card(self, index):
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
        high_list = filter(lambda x: len(x) > 1, number_values_in_hand)
        low_list = filter(lambda x: len(x) == 1, number_values_in_hand)
        return sorted(low_list) + sorted(high_list)

    def suits(self):
        """Return a list of the suits of every cars in a hand."""
        suits_in_hand = [self.hand[x].suit for x in range(self.current_hand_length())]
        return sorted(suits_in_hand)

    def duplicates(self, lst):
        """Detect duplicates within a provided list"""
        lst = sorted(lst)
        duplicates = []
        for x in range(len(lst) - 1):
            if lst[x] == lst[x + 1]:
                duplicates.append(lst[x])
        return duplicates


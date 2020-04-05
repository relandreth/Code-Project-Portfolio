"""
This Module contains the game aspect, querying the player to choose
which hole is more valuable with the community hand, telling them if they're right, etc.

@author: Roderick Landreth

Created October 9th 2018 for Card Project 3

I affirm that I have carried out my
academic endeavors with full academic honesty. Roderick Landreth
"""

import deck as dck
import stud_poker_hand as stud
import community_card_set as comm

CARDS_IN_HOLE = 2
CARDS_IN_FINAL_HAND = 5
MAXIMUM_ORDER_OF_CARD_MAGNITUDE = 14

class GuessingGame:
    def __init__(self, deck_in_use):
        self.deck = deck_in_use
        self.score = 0
        self.round_setup()

    def round_setup(self):
        """ Make new hands and start a Fresh game or round."""
        self.community_hand = comm.CommunityCardSet(self.deck)
        self.hole_hand_a = stud.StudPokerHand(self.deck)
        self.hole_hand_b = stud.StudPokerHand(self.deck)

        self.community_hand.draw_hand()
        self.hole_hand_a.draw_hand()
        self.hole_hand_b.draw_hand()

        self.total_cards_used = (2 * CARDS_IN_HOLE) + self.community_hand.cards_in_hand
        self.ask_for_best()

    def new_game(self):
        """Set up another game after one ends."""
        self.score = 0
        self.deck = dck.Deck()
        self.round_setup()

    def __best_hand(self):
        """Return which hole makes the most valuable hand with the community cards."""
        # I could use this as a different method in my poker_hand class to find the rank, as well.
        for x in range(self.community_hand.current_hand_length()):
            self.hole_hand_a.add_card(self.community_hand.return_card(x))
            self.hole_hand_b.add_card(self.community_hand.return_card(x))

        self.best_hand(self.hole_hand_a)
        self.best_hand(self.hole_hand_b)

        answer = self.hole_hand_a.compare_to(self.hole_hand_b)
        if answer == 1:
            return 'a'
        elif answer == -1:
            return 'b'
        elif answer == 0:
            return ' '

    def best_hand(self, big_hand):

        inventory = []
        scores_list = []
        for i in range(big_hand.current_hand_length()):
            for j in range(big_hand.current_hand_length()-1):
                disposable_hand = self.copy_hand(big_hand)
                disposable_hand.remove_card(i)
                disposable_hand.remove_card(j)
                inventory.append( (self.score_hand(disposable_hand), i, j) )
                scores_list.append( self.score_hand(disposable_hand) )
        best_hand = max(scores_list)
        for x in inventory:
            if x[0] == best_hand:
                i = x[1]
                j = x[-1]
                big_hand.remove_card(i)
                big_hand.remove_card(j)
                return big_hand

    def copy_hand(self,a_hand):
        """Create and return a hand that is a copy of the one provided"""
        new_hand = comm.CommunityCardSet(self.deck)
        for x in range(a_hand.current_hand_length()):
            new_hand.add_card(a_hand.return_card(x))
        return new_hand

    def list_mult(self, lst):
        """ Make each higher card value an order of magnitude
        more valuable, so a hand of 2's cannot beat a 3."""
        tot = 0
        for k in lst:
            tot += (10**int(k))
        return tot

    def score_hand(self,hand):
        """Score the hand based off its contents, pairs, and flush."""
        hand_value = self.list_mult(hand.numbers())
        pair_score = self.list_mult(set(hand.duplicates(hand.numbers())))
        suits = hand.suits()
        flush_score = 0
        flush = False
        for x in suits:
            if suits.count(x) >= CARDS_IN_FINAL_HAND:
                flush = True
        if flush:
            flush_score += 10**((MAXIMUM_ORDER_OF_CARD_MAGNITUDE+1)*2)
        final_score = hand_value + ((10**MAXIMUM_ORDER_OF_CARD_MAGNITUDE) * pair_score)+ flush_score
        return final_score

    def ask_for_best(self):
        """Query the player on their idea of which hole hand is better, and change score accordingly."""
        print("Which of the following holes is worth more in a five card hand?")
        print("{:^70}\n{:^20}\n\n{:>30}\n         {:>20}\n\n{:>30}\n          {:>20}\n".format \
                  ("Community Hand:", self.community_hand.__str__(), "Hole Hand A:", self.hole_hand_a.__str__(),
                   "Hold Hand B:", self.hole_hand_b.__str__()))
        answer = input(
            "Press 'a' or 'b' to indicate which makes a better hand-\n               (or Space if equal):    ")

        if str(answer[0]) == str(self.__best_hand()):
            self.score += 1
            print("||||||||||||||||||- {} -||||||||||||||||||".format("Correct!"))
            if self.deck.length() < self.total_cards_used:
                print("The deck ran out!")
                self.game_over()
            else:
                self.round_setup()

        else:
            print("||||||||||||||||||- {} -||||||||||||||||||\n The correct answer is '{}'".format("Incorrect", answer))
            self.game_over()

    def game_over(self):
        """End the game, show score, and ask if the player wants to play another."""
        print("Game Over. Score = {}".format(self.score))
        print()
        play_again = input("  -Play Again? (y/n)")
        if play_again.__contains__("y"):
            self.new_game()
        else:
            print("Goodbye!\n")

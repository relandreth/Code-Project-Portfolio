"""
This Module contains the game aspect, querying the player to choose
which hand won, telling them if they're right, etc.

@author: Roderick Landreth

Created October 9th 2018 for Card Project 3

I affirm that I have carried out my
academic endeavors with full academic honesty. Roderick Landreth
"""

import poker_hand as poker
import deck as deck


class SimpleGuessingGame:
    def __init__(self, deck):
        self.deck = deck
        self.score = 0
        self.game(self.deck,self.score)

    def game(self, remaining_deck, score):
        """Play a game determining which poker hand is the most valuable."""
        hand_a = poker.PokerHand(remaining_deck)
        hand_a.draw_hand()
        hand_b = poker.PokerHand(remaining_deck)
        hand_b.draw_hand()
        answer = hand_a.compare_to(hand_b)

        print("{:^70}\n{:^20}\n{:^70}\n{:^20}\n".format("Hand A:", hand_a.__str__(), "Hand B:", hand_b.__str__()))

        guess = self.__players_guess(score, answer)

        if guess and remaining_deck.length() >= 10:
            score += 1
            print("\n{:^60}\n".format("||||||||||||||You Are Correct!||||||||||||||"))
            self.game(remaining_deck, score)
        elif guess:
            score += 1
            print("\n{:^60}\n".format("||||||||||||||You Are Correct!||||||||||||||"))
            print("Sorry! Out of Cards!")
            self.__play_again(score)
        else:
            self.__play_again(score)

    def __players_guess(self, score, actual_outcome):
        """Queries the player for a guess, return if correct."""
        win_definition = {1: " Hand 1!",  -1:" Hand 2!", 0:" A PERFECT TIE! AHHHH (the crowd goes wild in the background"}
        play = input("\nWhich is the Winning Hand? (1 or 2, or tie?)  ")

        if play == str(actual_outcome) or (play == '2' and actual_outcome == -1) or (play == "tie" and actual_outcome == 0):
            return True
        else:
            print("||||||||||||||Incorrect.|||||||||||||| \n     The actual answer is:{}.".format(win_definition[actual_outcome]))
            return False

    def __play_again(self,score):
        """Tell the score, and ask if they want to play the game again."""
        play_again = input(" You Won: {:^10} times. Play again? (Y/N)     ".format(score))

        if str(play_again)[0] == "y":
            self.game(deck.Deck(), 0)
        elif str(play_again)[0] == "n":
            print("Fine. Goodbye!")
        else:
            print("I'm sorry, I couldn't understand that. What was that?")
            self.__play_again(score)


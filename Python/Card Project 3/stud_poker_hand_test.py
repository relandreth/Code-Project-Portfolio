"""
This module tests the stud_poker_hand module, specifically the __compare_to function.

@author: Roderick Landreth

Created October 9th 2018 for Card Project 3

I affirm that I have carried out my
academic endeavors with full academic honesty. Roderick Landreth
"""
import stud_poker_hand as stud
import deck as dck
import card as crd

def make_test_hand( lst ):
    """Make a test hand with a convenient list."""
    my_deck = dck.Deck()
    my_hand = stud.StudPokerHand(my_deck)
    for x in lst:
        card = crd.Card(x[0],x[1])
        my_hand.add_card(card)
    return my_hand


def compare_test(hand,hand2,outcome):
    """Compare two hand circumstances."""
    outcome_meaning = {1:"a wins",0:'Tie',-1:"b wins"}
    answer = hand.compare_to(hand2)
    print(hand.__str__())
    print(hand2.__str__())
    print("Expected outcome: {}".format(outcome_meaning[outcome]))
    if outcome == answer:
        print("PASS")
        print()
    else:
        print("FAIL")
        print("Instead: {}".format(outcome_meaning[answer]))
        print()

def main():
    # I Borrowed circumstances only from your testing from Project two, and gave you credit at the top of the module.

    print("2 pair vs 2 pair")
    winner = make_test_hand( [('D', 4), ('S', 7), ('D', 8), ('H', 7), ('S', 4)])
    loser = make_test_hand( [('D', 7), ('D', 10), ('S', 5), ('H', 12), ('C', 7)])
    compare_test(winner,loser,1)

    print("1 pair vs high card")
    winner = make_test_hand([('D', 6), ('D', 10), ('S', 10), ('H', 12), ('C', 2)])
    loser = make_test_hand([('D', 13), ('S', 14), ('D', 8), ('H', 7), ('S', 4)])
    compare_test(winner,loser,1)

    print("flush vs high card")
    winner = make_test_hand([('H', 4), ('H', 7), ('H', 8), ('H', 11), ('H', 2)])
    loser = make_test_hand( [('D', 7), ('D', 10), ('S', 5), ('H', 12), ('C', 7)])
    compare_test(winner,loser,1)

    print("2 pair vs high card")
    winner = make_test_hand([('D', 6), ('D', 10), ('S', 10), ('H', 12), ('C', 12)])
    loser = make_test_hand([('D', 13), ('S', 14), ('D', 8), ('H', 7), ('S', 4)])
    compare_test(winner,loser,1)

    print("flush vs two pair")
    winner = make_test_hand([('S', 4), ('S', 7), ('S', 8), ('S', 11), ('S', 2)])
    loser = make_test_hand( [('D', 7), ('D', 10), ('H', 7), ('H', 10), ('D', 14)])
    compare_test(winner,loser,1)

    print("flush vs high card")
    winner = make_test_hand([('D', 6), ('D', 10), ('D', 3), ('D', 12), ('D', 14)])
    loser = make_test_hand( [('D', 13), ('S', 14), ('D', 8), ('H', 7), ('S', 4)])
    compare_test(winner,loser,1)

    print("two flushes different on third high card")
    winner = make_test_hand([('D', 11), ('D', 7), ('D', 2), ('D', 9), ('D', 3)])
    loser = make_test_hand( [('H', 2), ('H', 9), ('H', 5), ('H', 3), ('H', 11)])
    compare_test(winner,loser,1)

    print("two flushes pefectly tied")
    winner = make_test_hand([('D', 13), ('D', 7), ('D', 3), ('D', 9), ('D', 5)])
    loser = make_test_hand( [('H', 13), ('H', 9), ('H', 5), ('H', 7), ('H', 3)])
    compare_test(winner,loser,0)

    print("two 2 pair hands different on low pairs")
    winner = make_test_hand([('D', 8), ('H', 2), ('C', 10), ('D', 10), ('H', 8)])
    loser = make_test_hand( [('S', 6), ('H', 6), ('H', 5), ('H', 10), ('S', 10)])
    compare_test(winner,loser,1)

    print("two 1 pair hands different on pair")
    winner = make_test_hand([('S', 12), ('H', 11), ('H', 12), ('C', 2), ('S', 4)])
    loser= make_test_hand( [('D', 12), ('C', 11), ('C', 14), ('D', 11), ('H', 13)])
    compare_test(winner, loser, 1)

    print("two high card hands different on last card")
    winner = make_test_hand( [('C', 6), ('D', 7), ('H', 11), ('S', 13), ('D', 12)])
    loser = make_test_hand([('H', 7), ('C', 13), ('H', 12), ('D', 11), ('S', 5)])




if __name__ == "__main__":
    main()

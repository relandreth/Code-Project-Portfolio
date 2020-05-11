/**This class creates a hand of cards and manipulates it.
 *
 * @author Roderick Landreth
 * Created: November 9, 2018
 *
I affirm that I have carried out my
academic endeavors with full academic honesty. Roderick Landreth
"""
 */
import java.util.ArrayList;
import java.util.Collections;
import java.lang.Math;

public class PokerHand {
    private int cardsInFlush = 5;
    private int maximumOrderOfCardMagnitude = 14;
    private ArrayList<Card> hand = new ArrayList<>();

    public PokerHand(){
    }

    /**Produce a string in the format of which the cards in the hand would be read aloud.
     *
     * @return a string containing information on all cards in the hand
     */
    public String toString(){
        String handString = "";
        for (Card card : hand){
            handString += card.toString() + ", ";
        }
        return handString;
    }

    /**Add a given card to this hand.
     *
     * @param card A given card to add to the hand
     */
    public void addCard(Card card){
        hand.add(card);
    }

    /**Return a card and its information from a hand
     *
     * @param index the order of the card in the hand
     * @return a copy of the card at that index of the hand, to prevent that object from being changed.
     */
    public Card returnCard(int index){
        return hand.get(index).copy();
    }

    /**Remove the card at a given index from the hand.
     *
     * @param index the order of the card in the hand
     */
    public void removeCard(int index){
        hand.remove(index);
    }

    /**Check to see if a hand contains a certain card or not.
     *
     * @param thisCard a supplied card
     * @return whether thisCard is within this hand.
     */
    public boolean contains(Card thisCard){
        return hand.contains(thisCard);
    }

    /**Check how many cards the hand currently contains.
     *
     * @return an int of the size of the current hand.
     */
    public int currentHandLength(){
        return hand.size();
    }

    /**Create a list containing the sorted values within a hand of cards.
     *
     * @return all values in the hand.
     */
    public ArrayList<Integer> numbers(){
        ArrayList<Integer> handValues = new ArrayList<>(0);
        for (Card card : hand){
            handValues.add(card.getRank());
        }
        Collections.sort(handValues);
        return handValues;
    }

    /**Create a list containing the sorted suits within a hand of cards.
     *
     * @return all suits in the hand.
     */
    public ArrayList<Character> suits(){
        ArrayList<Character> handSuits = new ArrayList<Character>(0);
        for (Card card : hand){
            handSuits.add(card.getSuit());
        }
        Collections.sort(handSuits);
        return handSuits;
    }

    /**Tell the values that are repeated within a given hand.
     *
     * @return an ArrayList containing the amount of duplicate card types.
     */
    public ArrayList<Integer> duplicates(){
        ArrayList<Integer> duplicateCards = new ArrayList<>(0);
        for (int i=0; i<hand.size()-1; i++){
            if (hand.get(i).getRank() == hand.get(i+1).getRank() && !duplicateCards.contains(hand.get(i).getRank()) ){
                duplicateCards.add(hand.get(i).getRank());
            }
        }
        return duplicateCards;
    }

    /**Assign a number to the hand that encapsulates each card value, pairs and pair values, and if there is a flush in
     * increasing magnitudes.
     *
     * @return the 'score' of each hand.
     */
    public double scoreHand(){
        double handScore=0;
        double pairScore=0;
        double flushScore=0;
        ArrayList<Integer> cardValues = numbers();
        ArrayList<Integer> pairs = duplicates();
        for(Integer number : cardValues){
            handScore += Math.pow(10, number-1);
        }

        if (pairs.size() >= 1){
            pairScore += Math.pow(2,pairs.get(0)+maximumOrderOfCardMagnitude);
            if (pairs.size() >= 2){
                pairScore += Math.pow(2,pairs.get(1)+maximumOrderOfCardMagnitude+1);
            }
        }
        if (isFlush()){
            flushScore = Math.pow(2,(maximumOrderOfCardMagnitude*2));
        }
        return handScore + pairScore + flushScore;
    }

    /**Return true if there is a flush within this hand.
     *
     * @return true if the sorted suits array has chars with an index difference of 4 that are the same.
     */
    private boolean isFlush(){
        ArrayList suits = suits();
        boolean fiveSimilarSuits = false;
        if (suits.size()>=5) {
            for (int i = 0; i <= suits.size() - cardsInFlush + 1; i++) {
                if (suits.get(i) == suits.get(i + cardsInFlush - 1)) {
                    fiveSimilarSuits = true;
                }
            }
        }
        return fiveSimilarSuits;
    }

    /**
     *  Determine how this hand compares to another hand, returns
     *  positive, negative, or zero depending on the comparison.
     *
     *  @param other The hand to compare this hand to
     *  @return a negative number if this is worth LESS than other, zero
     *  if they are worth the SAME, and a positive number if this is worth
     *  MORE than other
     */
    public int compareTo(PokerHand other){
        double thisHand = scoreHand();
        double thatHand = other.scoreHand();
        return Double.compare(thisHand, thatHand);
    }
}

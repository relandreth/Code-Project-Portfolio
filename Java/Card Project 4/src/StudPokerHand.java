public class StudPokerHand extends PokerHand {

    private int cardsInHand=5;

    public StudPokerHand(){
    }

    /**Copy the input hand, making an identical copy of it for information hiding and manipulation.
     *
     * @param hand the hand that needs to be copied
     * @return an identical hand that is a separate object from the original.
     */
    public StudPokerHand copy(StudPokerHand hand){
        StudPokerHand newHand = new StudPokerHand();
        for (int i=0; i<hand.currentHandLength(); i++){
            newHand.addCard(returnCard(i).copy());
        }
        return newHand;
    }

    /**
     *  Determine how this hand compares to another hand, and return
     *  positive, negative, or zero depending on the comparison.
     *
     *  @param other The hand to compare this hand to
     *  @return a negative number if this is worth LESS than other, zero
     *  if they are worth the SAME, and a positive number if this is worth
     *  MORE than other
     */
    public int compareTo(StudPokerHand other){
        double thisHand = scoreHand();
        double thatHand = other.scoreHand();
        return Double.compare(thisHand, thatHand);
    }

}
public class PokerComparisonTests {

    public static void main (String [] args)
    {
        Tester t = new Tester(true);

        testBothFlush(t);
        // ... call more test methods here
        // ... write those method below

        t.finishTests();
    }

    private static PokerHand stringToPokerHand(String[] cardsAsStringsList) {
        PokerHand hand = new PokerHand();
        for (String cardDescriptor : cardsAsStringsList) {
            char suit = cardDescriptor.charAt(0);
            int value = Integer.parseInt(cardDescriptor.substring(1));
            hand.addCard(new Card(suit, value));
        }
        return hand;
    }

    private static void testBothFlush(Tester t) {
        String msg = "Hand A and B are both a flush. A should win.";
        PokerHand handA = stringToPokerHand(new String[]{"H2", "H14", "H11", "H10", "H9"});
        PokerHand handB = stringToPokerHand(new String[]{"H2", "H14", "H11", "H8", "H9"});
        t.assertEquals(msg, true, handA.compareTo(handB) > 0);
    }

}

/**This class creates cards and shuffles a deck of cards.
 *
 * @author Roderick Landreth
 * Created: November 9, 2018
 *
I affirm that I have carried out my
academic endeavors with full academic honesty. Roderick Landreth
"""
 */
import java.util.ArrayList;
//import java.util.Arrays;
import java.util.Random;

public class Deck {
    private int maxCardVal = 14;
    private char[] possibleSuits = {'H','S','C','D'};
    Random rand = new Random();
    ArrayList<Card> deckArray = new ArrayList<>();

    /**
     *Fill deck with 52 unique cards.
     */
    public Deck(){
        for(int i=0; i<maxCardVal; i++){
            for(char suit : possibleSuits){
                Card newCard = new Card(suit,i+1);
                deckArray.add(newCard);
            }
        }
    }

    /**deal one copy of a card from the deck, removing it from the deck completely.
     *
     * @return card dealt by deck
     */
    public Card deal(){
        Card card = deckArray.get(rand.nextInt(deckArray.size()));
        deckArray.remove(card);
        return card.copy();
    }

    /**Check the size of the deck.
     *
     * @return size of deck
     */
    public int size(){
        return deckArray.size();
    }

    /*public static void main(){
        Deck myDeck = new Deck();
        System.out.println(myDeck.size());
        Card aCard = myDeck.deal();
        System.out.println(aCard);
        Card bCard = myDeck.deal();
        System.out.println(bCard);
        Card cCard = myDeck.deal();
        System.out.println(cCard);
        System.out.println(myDeck.size());
    }*/
}

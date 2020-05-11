/**This class creates cards and dispenses information about a card object.
 *
 * @author Roderick Landreth
 * Created: November 9, 2018
 *
 I affirm that I have carried out my
 academic endeavors with full academic honesty. Roderick Landreth
 """
*/
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Dictionary;


public class Card {
    private ArrayList<String> VALUES = new ArrayList<String>(
            Arrays.asList("2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King", "Ace") );
    //private Dictionary SUITS.put('H', "Poop");
    private char Suits;
    private int Value;

    public Card(char Suits, int Value){
        this.Suits = Suits;
        this.Value = Value;
    }

    /**Return the value of a given card.
     *
     * @return card's value
     */
    public int getRank(){
        return Value;
    }

    /**Return the suit of a given card.
     *
     * @return card's suit
     */
    public char getSuit(){
        return Suits;
    }

    /**Provide a string in the format of a card's name.
     *
     * @return a formatted string of the card representation.
     */
    public String toString(){
    return VALUES.get(Value-2)+" of " + Suits;
    }

    /**Create a copy of the card and return it, so when a card is returned the deck or hand cannot be altered
     * outside the program. Both instance variables are primitives, so a copy of their value in memory is returned.
     *
     * @return a copy of the card in question, for information hiding purposes.
     */
    public Card copy(){
        return new Card(Suits, Value);
    }

    /*public static void main(){
        Card newCard = new Card('S',3);
        System.out.println(newCard.getSuit());
        System.out.println(newCard.getRank());
        System.out.println(newCard.toString());
    }*/
}

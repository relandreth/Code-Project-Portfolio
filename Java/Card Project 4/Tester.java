/**
 * This class contains a collection of methods that help with testing.
 *
 * @author Chris Fernandes, Aaron Cass, Kristina Striegnitz
 */
public class Tester {
    private int numPasses;
    private int numFails;
    private boolean verbose;

    /**
     * Creates a tester.
     *
     * @param verbose an indication of whether verbose output is to be
     *                printed.
     */
    public Tester(boolean verbose) {
        this.numPasses = 0;
        this.numFails = 0;
        this.verbose = verbose;
    }

    /**
     * Creates a tester with verbose output.
     */
    public Tester() {
        this(true);
    }

    /**
     * Toggles between a lot of output and little output.
     *
     * @param verbose If verbose is true, then complete information is printed,
     *                whether the tests passes or fails. If verbose is false, only
     *                failures are printed.
     */
    public void setVerbose(boolean verbose) {
        this.verbose = verbose;
    }

    /**
     * Each of the assertEquals methods tests whether the actual result equals
     * the expected result. If it does, then the test passes, otherwise it
     * fails.
     * <p>
     * The only difference between these methods is the types of the
     * parameters.
     * <p>
     * All take a String message and two values of the same type (either
     * boolean, int, or Object) to compare:
     *
     * @param message  a message or description of the test
     * @param expected the correct, or expected, value
     * @param actual   the actual value
     */
    public void assertEquals(String message, boolean expected, boolean actual) {
        printTestCaseInfo(message, "" + expected, "" + actual);
        if (expected == actual) {
            pass();
        } else {
            fail(message);
        }
    }

    public void assertEquals(String message, int expected, int actual) {
        printTestCaseInfo(message, "" + expected, "" + actual);
        if (expected == actual) {
            pass();
        } else {
            fail(message);
        }
    }

    public void assertEquals(String message, Object expected, Object actual) {
        printTestCaseInfo(message, "" + expected, "" + actual);

        if (expected == null) {
            if (actual == null) {
                pass();
            } else {
                fail(message);
            }
        } else if (expected.equals(actual)) {
            pass();
        } else {
            fail(message);
        }

    }

    private void printTestCaseInfo(String message, String expected,
                                   String actual) {
        if (verbose) {
            System.out.println(message + ":");
            System.out.println("expected: " + expected);
            System.out.println("actual:   " + actual);
        }
    }

    private void pass() {
        numPasses++;
        if (verbose) {
            System.out.println("PASS");
            System.out.println();
        }
    }

    private void fail(String description) {
        numFails++;

        if (!verbose) {
            System.out.print(description + "  ");
        }
        System.out.println("*******########## FAIL");
        System.out.println();
    }

    /**
     * Prints a header for a section of tests.
     *
     * @param sectionTitle The header that should be printed.
     */
    public void testSection(String sectionTitle) {
        if (verbose) {
            int dashCount = sectionTitle.length();
            System.out.println(sectionTitle);
            for (int i = 0; i < dashCount; i++) {
                System.out.print("-");
            }
            System.out.println();
            System.out.println();
        }
    }

    /**
     * Prints summary statistics after the tests are complete.
     */
    public void finishTests() {
        int totalTests = numPasses + numFails;
        System.out.println("Passed " + numPasses + "/" + totalTests);
        System.out.println("Failed " + numFails + "/" + totalTests);
    }
}

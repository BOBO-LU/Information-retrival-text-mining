# Import package
from nltk.stem import PorterStemmer


class PA1:

    def __init__(self, txt):
        '''Initialize all the objects on class instantiation.'''
        self.txt = txt
        self.tokens = []

    def tokenization(self):
        '''Tokenize tokens in PA1.'''
        valid = ""

        # Save alphabets and space into a new string called valid
        for character in self.txt:
            if character.isalpha() or character.isspace():
                valid += (character)

        # Split the string "valid" by space into list
        self.tokens = valid.split()

    def lowercasing(self):
        '''Lowercase tokens in PA1.'''

        # For every token inside "tokens", lowercase it
        for idx, token in enumerate(self.tokens):
            self.tokens[idx] = token.lower()

    def steamming_with_porter(self):
        '''Steamming tokens in PA1 with porter algorithm.'''

        # Initialzie porter algorithm as "ps"
        ps = PorterStemmer()

        # For every token inside "tokens", use "ps" to stem it
        for idx, token in enumerate(self.tokens):
            self.tokens[idx] = ps.stem(token)

    def stopword_removal(self):
        '''Remove all stopwords in tokens in PA1.'''
        clean_tokens = []

        # Declare the stopwords
        STOP_WORDS = ['were', 'shan', 'them', 'i', 'just', 'him', 'below', 'both', "you'd",
                      'don', 'wouldn', "wouldn't", 'then', 's', 'will', 'how', 'wasn', 'am',
                      'should', 'from', 'hasn', 'each', 'any', 'yours', 'who', 'such', 'can',
                      'once', 'on', 'all', 'haven', 'didn', 'again', 'ain', 'doesn', 'same',
                      't', 'and', 'hers', "weren't", 'until', 'has', 'themselves', 'in', 'she',
                      "you've", "couldn't", 'a', 'about', 'been', 'because', 'herself', 'ourselves',
                      'with', 'isn', "didn't", 'mustn', 'needn', 'or', "you're", 'but', 'you',
                      "mightn't", 'of', 'under', 'where', "that'll", 'which', 'does', "hasn't",
                      'have', 'mightn', "hadn't", 'y', 'what', 'won', 'he', 'nor', 'between',
                      'couldn', 'an', 'whom', 'than', 'no', 'd', 'yourself', 'only', 'the', 'this',
                      'after', 'her', "shan't", 'itself', 'being', 'do', 'against', 'into',
                      'me', 'over', "haven't", "don't", 'your', 'now', 'we', 'aren', 'some',
                      'why', 'very', 'shouldn', 'ours', 'doing', 'ma', 'off', 'there', 'himself',
                      "isn't", 'at', 'during', 'had', 'too', 'my', 'before', 'it', 'while', 'most',
                      "doesn't", 'few', 'be', 'hadn', 'those', 'theirs', 'its', 'here', 'll',
                      "you'll", 'myself', 'further', 're', 'their', 'they', "should've", "it's",
                      'by', 'his', 'are', 'yourselves', 'through', 've', 'above', 'o', 'when',
                      'so', "shouldn't", 'out', 'm', "aren't", 'these', 'not', 'weren', 'did',
                      'own', 'if', 'is', 'having', 'as', "mustn't", "won't", 'up', 'our', "she's",
                      'was', 'down', "needn't", 'more', 'that', "wasn't", 'to', 'other', 'for']

        # Append token from "tokens" which does not exist in "STOP_WORDS" into a new list
        for token in self.tokens:
            if token not in STOP_WORDS:
                clean_tokens.append(token)

        # Replace the tokens in PA1 with the one without any stopwords
        self.tokens = clean_tokens

    def save_result(self):
        '''Save the tokens in PA1 into text file named result.txt.'''

        # Open a textfile and write the tokens into it
        with open("result.txt", "w") as f:
            data = " ".join(self.tokens)
            f.write(data)

    def print_tokens(self):
        '''Print tokens in PA1.'''

        # Print the object PA1's tokens
        print(self.tokens)


if __name__ == "__main__":

    # Declare the corpus to process
    CORPUS = """And Yugoslav authorities are planning the arrest of eleven coal miners 
    and two opposition politicians on suspicion of sabotage, that's in 
    connection with strike action against President Slobodan Milosevic. 
    You are listening to BBC news for The World."""

    pa1 = PA1(CORPUS)  # Create a PA1 instance.
    pa1.tokenization()  # Tokenization.
    pa1.lowercasing()  # Lowercasing everything.
    pa1.steamming_with_porter()  # Stemming using Porter’s algorithm.
    pa1.stopword_removal()  # Stopword removal.
    pa1.save_result()  # Save the result as a txt file.
    pa1.print_tokens()  # Print tokens.

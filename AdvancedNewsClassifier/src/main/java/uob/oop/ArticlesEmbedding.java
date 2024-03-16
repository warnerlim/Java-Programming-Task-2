package uob.oop;

import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.pipeline.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import java.util.Properties;

public class ArticlesEmbedding extends NewsArticles {
    private int intSize = -1;
    private String processedText = "";
    private INDArray newsEmbedding = Nd4j.create(0);

    public ArticlesEmbedding(String _title, String _content, NewsArticles.DataType _type, String _label) {
        //TODO Task 5.1 - 1 Mark
        super( _title, _content, _type, _label);
    }

    public void setEmbeddingSize(int _size) {
        //TODO Task 5.2 - 0.5 Marks
        intSize = _size;
    }

    public int getEmbeddingSize(){
        return intSize;
    }

    @Override
    public String getNewsContent() {
        if (!processedText.isEmpty()) {
            return processedText.toLowerCase().trim();
        }
        // TODO Task 5.3 - 10 Marks
        String pre_processed_string = super.getNewsContent();
        String cleaned_text = textCleaning(pre_processed_string);
        String lemmatized_text = lemmatizeText(cleaned_text).toLowerCase();
        processedText = removeStopWords(lemmatized_text, Toolkit.STOPWORDS);

        return processedText.toLowerCase().trim();
    }

    public INDArray getEmbedding() throws Exception {
        // TODO Task 5.4 - 20 Marks
        if (!newsEmbedding.isEmpty()) return Nd4j.vstack(newsEmbedding.mean(1));

        int max_vector_size = 0;
        for (Glove largest_size : AdvancedNewsClassifier.listGlove) {
            if (largest_size != null) {
                double[] double_array = largest_size.getVector().getAllElements();
                int currentSize = double_array.length;

                if (currentSize > max_vector_size) {
                    max_vector_size = currentSize;
                }
            }
        }

        if (intSize == -1) {
            throw new InvalidSizeException("Invalid size");
        }

        if (processedText.isEmpty()) {
            throw new InvalidTextException("Invalid text");
        }

        String[] words = processedText.split(" ");

        newsEmbedding = Nd4j.create(intSize, max_vector_size);
        int inserted_words = 0;

        for (int i = 0; i < words.length && inserted_words < intSize; i++) {
            String currentWord = words[i];
            // Find the WordArray object with the current word
            Glove wordVector = findGloveObject(currentWord);
            // Check if wordVector is not null before inserting into newsEmbedding
            if (wordVector != null) {
                // Get the vector associated with the word
                double[] doubleArray = wordVector.getVector().getAllElements();
                // Put the vector as a row in the ND4J array
                newsEmbedding.putRow(inserted_words, Nd4j.create(doubleArray));
                inserted_words++;
            }
        }

        return Nd4j.vstack(newsEmbedding.mean(1));
    }

    private Glove findGloveObject(String word) {
        for (Glove glove : AdvancedNewsClassifier.listGlove) {
            if (glove.getVocabulary().equals(word)) {
                return glove;
            }
        }
        return null;
    }

    /***
     * Clean the given (_content) text by removing all the characters that are not 'a'-'z', '0'-'9' and white space.
     * @param _content Text that need to be cleaned.
     * @return The cleaned text.
     */
    private static String textCleaning(String _content) {
        StringBuilder sbContent = new StringBuilder();

        for (char c : _content.toLowerCase().toCharArray()) {
            if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || Character.isWhitespace(c)) {
                sbContent.append(c);
            }
        }

        return sbContent.toString().trim();
    }
    private String lemmatizeText(String text) {

        Properties props = new Properties();
        props.setProperty("annotators", "tokenize,pos,lemma");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
        CoreDocument document = pipeline.processToCoreDocument(text);
        StringBuilder lemmatizedText = new StringBuilder();
        for (CoreLabel tok : document.tokens()) {
            lemmatizedText.append(tok.lemma()).append(" ");
        }
        return lemmatizedText.toString().trim();
    }
    public static String removeStopWords(String _content, String[] _stopWords) {
        StringBuilder mySB = new StringBuilder();
        String[] wordsList = _content.split(" ");
        for (String word : wordsList) {
            if (notContains(_stopWords, word)) {
                mySB.append(word).append(" ");
            }
        }

        return mySB.toString().trim();
    }
    private static boolean notContains(String[] _arrayTarget, String _searchValue) {
        for (String element : _arrayTarget) {
            if (_searchValue.equals(element)) {
                return false;
            }
        }
        return true;
    }
}



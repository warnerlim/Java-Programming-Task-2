package uob.oop;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Stream;

public class Toolkit {
    public static List<String> listVocabulary = null;
    public static List<double[]> listVectors = null;
    private static final String FILENAME_GLOVE = "glove.6B.50d_Reduced.csv";

    public static final String[] STOPWORDS = {"a", "able", "about", "across", "after", "all", "almost", "also", "am", "among", "an", "and", "any", "are", "as", "at", "be", "because", "been", "but", "by", "can", "cannot", "could", "dear", "did", "do", "does", "either", "else", "ever", "every", "for", "from", "get", "got", "had", "has", "have", "he", "her", "hers", "him", "his", "how", "however", "i", "if", "in", "into", "is", "it", "its", "just", "least", "let", "like", "likely", "may", "me", "might", "most", "must", "my", "neither", "no", "nor", "not", "of", "off", "often", "on", "only", "or", "other", "our", "own", "rather", "said", "say", "says", "she", "should", "since", "so", "some", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "tis", "to", "too", "twas", "us", "wants", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "would", "yet", "you", "your"};

    public void loadGlove() throws IOException {
        BufferedReader myReader = null;
        //TODO Task 4.1 - 5 marks
        try {
            listVocabulary = new ArrayList<>();
            listVectors = new ArrayList<>();
            myReader = new BufferedReader(new FileReader(Toolkit.getFileFromResource(FILENAME_GLOVE)));
            String resultLine = myReader.readLine();

            while (resultLine != null) {
                String[] splitResult = resultLine.split(",");
                listVocabulary.add(splitResult[0]);
                double[] doubVectors = new double[splitResult.length - 1];
                for (int i = 1; i < splitResult.length; i++) {
                    doubVectors[i - 1] = Double.parseDouble(splitResult[i]);
                }
                listVectors.add(doubVectors);
                resultLine = myReader.readLine();
            }
        } catch (Exception e) {
            System.out.println(e.getMessage());
        } finally {
            myReader.close();
        }
    }

    private static File getFileFromResource(String fileName) throws URISyntaxException {
        ClassLoader classLoader = Toolkit.class.getClassLoader();
        URL resource = classLoader.getResource(fileName);
        if (resource == null) {
            throw new IllegalArgumentException(fileName);
        } else {
            return new File(resource.toURI());
        }
    }

    public List<NewsArticles> loadNews() {
        List<NewsArticles> listNews = new ArrayList<>();
        //TODO Task 4.2 - 5 Marks
        try (Stream<Path> paths = Files.walk(Paths.get("src/main/resources/News"))) {
            paths.filter(Files::isRegularFile).filter(p -> p.toString().endsWith(".htm")).sorted(Comparator.comparing(Path::getFileName)).forEach(p -> {
                StringBuilder content = new StringBuilder();
                try (BufferedReader br = Files.newBufferedReader(p)) {
                    String line;
                    while ((line = br.readLine()) != null) {
                        content.append(line).append("\n");
                    }
                    NewsArticles myNews = new NewsArticles(HtmlParser.getNewsTitle(content.toString()), HtmlParser.getNewsContent(content.toString()), HtmlParser.getDataType(content.toString()), HtmlParser.getLabel(content.toString()));
                    listNews.add(myNews);
                } catch (IOException e) {
                    System.out.println(e.getMessage());
                }
            });
        } catch (IOException e) {
            System.out.println(e.getMessage());
        }

        return listNews;
    }

    public static List<String> getListVocabulary() {
        return listVocabulary;
    }

    public static List<double[]> getlistVectors() {
        return listVectors;
    }
}

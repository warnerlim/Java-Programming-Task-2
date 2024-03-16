package uob.oop;

import org.apache.commons.lang3.time.StopWatch;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static uob.oop.Toolkit.*;

public class AdvancedNewsClassifier {
    public Toolkit myTK = null;
    public static List<NewsArticles> listNews = null;
    public static List<Glove> listGlove = null;
    public List<ArticlesEmbedding> listEmbedding = null;
    public MultiLayerNetwork myNeuralNetwork = null;

    public final int BATCHSIZE = 10;

    public int embeddingSize = 0;
    private static StopWatch mySW = new StopWatch();

    public AdvancedNewsClassifier() throws IOException {
        myTK = new Toolkit();
        myTK.loadGlove();
        listNews = myTK.loadNews();
        listGlove = createGloveList();
        listEmbedding = loadData();
    }

    public static void main(String[] args) throws Exception {
        mySW.start();
        AdvancedNewsClassifier myANC = new AdvancedNewsClassifier();

        myANC.embeddingSize = myANC.calculateEmbeddingSize(myANC.listEmbedding);
        myANC.populateEmbedding();
        myANC.myNeuralNetwork = myANC.buildNeuralNetwork(2);
        myANC.predictResult(myANC.listEmbedding);
        myANC.printResults();
        mySW.stop();
        System.out.println("Total elapsed time: " + mySW.getTime());
    }

    public List<Glove> createGloveList() {
        List<Glove> listResult = new ArrayList<>();
        List<String> vocabulary_list = listVocabulary;
        List<double[]> vector_list = listVectors;
        //TODO Task 6.1 - 5 Marks
        for (int i = 0; i < vocabulary_list.size(); i++) {
            String vocabulary = vocabulary_list.get(i);

            // Check if the vocabulary is a non-stop word
            if (!isStopWord(vocabulary)) {
                double[] vectorArray = vector_list.get(i);
                Vector vector = new Vector(vectorArray);
                Glove glove = new Glove(vocabulary, vector);
                listResult.add(glove);
            }
        }
        return listResult;
    }

    private boolean isStopWord(String word) {
        for (String stopword : STOPWORDS) {
            if (stopword.equalsIgnoreCase(word)) {
                return true;
            }
        }
        return false;
    }

    public static List<ArticlesEmbedding> loadData() {
        List<ArticlesEmbedding> listEmbedding = new ArrayList<>();
        for (NewsArticles news : listNews) {
            ArticlesEmbedding myAE = new ArticlesEmbedding(news.getNewsTitle(), news.getNewsContent(), news.getNewsType(), news.getNewsLabel());
            listEmbedding.add(myAE);
        }
        return listEmbedding;
    }

    public int calculateEmbeddingSize(List<ArticlesEmbedding> _listEmbedding) {
        int intMedian;
        List<Integer> document_lengths = new ArrayList<>();

        //TODO Task 6.2 - 5 Marks

        for (ArticlesEmbedding embedding : _listEmbedding) {
            String[] unprocessed = embedding.getNewsContent().split(" ");
            int word_count = 0;
            for (String currentWord : unprocessed) {
                for (Glove glove : listGlove) {
                    if (glove.getVocabulary().equals(currentWord)) {
                        word_count++;
                        break; // Break once the word is found in the Glove list
                    }
                }
            }
            document_lengths.add(word_count);
        }
        document_lengths.sort(null);
        intMedian = calculateMedian(document_lengths);
        return intMedian;
    }

    public static int calculateMedian(List<Integer> sortedList) {
        int size = sortedList.size();
        if (size % 2 == 0) {
            int middle1 = sortedList.get(size / 2);
            int middle2 = sortedList.get(size / 2 + 1);
            return (middle1 + middle2) / 2;
        } else {
            return sortedList.get((size + 1) / 2);
        }
    }

    public void populateEmbedding() {
        //TODO Task 6.3 - 10 Marks
        for (int i = 0; i < listEmbedding.size(); i++) {
            try {
                listEmbedding.get(i).setEmbeddingSize(embeddingSize);
                listEmbedding.get(i).getEmbedding();
            } catch (Exception ex) {
                if (ex.getClass().equals(InvalidSizeException.class)) {
                    listEmbedding.get(i).setEmbeddingSize(embeddingSize);
                } else if (ex.getClass().equals(InvalidTextException.class)) {
                    listEmbedding.get(i).getNewsContent();
                }
                i--;
            }
        }
    }

    public DataSetIterator populateRecordReaders(int _numberOfClasses) throws Exception {
        ListDataSetIterator myDataIterator = null;
        List<DataSet> listDS = new ArrayList<>();
        INDArray inputNDArray = null;
        INDArray outputNDArray = null;

        //TODO Task 6.4 - 8 Marks
        for (ArticlesEmbedding embedding : listEmbedding) {
            if (embedding.getNewsType() == NewsArticles.DataType.Training) {
                inputNDArray = embedding.getEmbedding();
                outputNDArray = Nd4j.create(1, _numberOfClasses);
                outputNDArray.putScalar(Integer.parseInt(embedding.getNewsLabel()) - 1, 1);
                DataSet myDataSet = new DataSet(inputNDArray, outputNDArray);
                listDS.add(myDataSet);
            }
        }

        return new ListDataSetIterator(listDS, BATCHSIZE);
    }


    public MultiLayerNetwork buildNeuralNetwork(int _numOfClasses) throws Exception {
        DataSetIterator trainIter = populateRecordReaders(_numOfClasses);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(42)
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .updater(Adam.builder().learningRate(0.02).beta1(0.9).beta2(0.999).build())
                .l2(1e-4)
                .list()
                .layer(new DenseLayer.Builder().nIn(embeddingSize).nOut(15)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.HINGE)
                        .activation(Activation.SOFTMAX)
                        .nIn(15).nOut(_numOfClasses).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        for (int n = 0; n < 100; n++) {
            model.fit(trainIter);
            trainIter.reset();
        }
        return model;
    }

    public List<Integer> predictResult(List<ArticlesEmbedding> _listEmbedding) throws Exception {
        List<Integer> listResult = new ArrayList<>();
        //TODO Task 6.5 - 8 Marks
        for (ArticlesEmbedding articlesEmbedding : _listEmbedding) {
            String dataType = String.valueOf(articlesEmbedding.getNewsType());
            if (dataType.equals("Testing")) {
                INDArray inputNDArray = articlesEmbedding.getEmbedding();
                int predictedLabel = myNeuralNetwork.predict(inputNDArray)[0];
                articlesEmbedding.setNewsLabel(String.valueOf(predictedLabel));
                listResult.add(predictedLabel);
            }
        }

        return listResult;
    }

    public void printResults() {
        //TODO Task 6.6 - 6.5 Marks
        int maxLabelValue = 0;
        for (ArticlesEmbedding articlesEmbedding : listEmbedding) {
            String dataType = String.valueOf(articlesEmbedding.getNewsType());
            if (dataType.equals("Testing")) {
                int labelValue = Integer.parseInt(articlesEmbedding.getNewsLabel());
                maxLabelValue = Math.max(maxLabelValue, labelValue);
            }
        }
        List<List<String>> labelGroups = new ArrayList<>(maxLabelValue);

        for (int i = 0; i <= maxLabelValue; i++) {
            labelGroups.add(new ArrayList<>());
        }

        for (ArticlesEmbedding articlesEmbedding : listEmbedding) {
            String dataType = String.valueOf(articlesEmbedding.getNewsType());
            if (dataType.equals("Testing")) {
                String newsTitle = articlesEmbedding.getNewsTitle();
                int labelValue = Integer.parseInt(articlesEmbedding.getNewsLabel());

                // Add news title to the corresponding list based on label value
                labelGroups.get(labelValue).add(newsTitle);
            }
        }
        for (int i = 0; i <= maxLabelValue; i++) {
            System.out.println("Group " + (i + 1));
            for (String newsTitle : labelGroups.get(i)) {
                System.out.println(newsTitle);
            }
        }
    }
}

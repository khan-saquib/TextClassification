import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.functions.Logistic;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;

/**
 *
 * @author Saquib
 */
public class NNWeka {

    boolean stopListEnable;
    List<String> stopWords;
    List<String> vocabularyList;
    HashMap<String, Integer> vocabularyListMap;
    ArrayList<Map<Integer, Integer>> documentsAll;
    ArrayList<Integer> documentClass;
    ArrayList<String> classValueString;
    HashMap<String, Integer> testClassValues;
    ArrayList<Integer> testDocumentClass;
    ArrayList<String> testFileNames;

    public static String TRAINFOLDERNAME;
    public static List<String> ALLFOLDERNAMES_TRAINING;
    public static String TESTFOLDERNAME;
    public static String STOPWORDS;
    public static String CLASSNAMES;
    public static String DEV_LABEL;
    public static String OUTPUT_FILE;

    public NNWeka(String[] args, boolean stopListEnable) throws FileNotFoundException {
        documentsAll = new ArrayList<>();
        documentClass = new ArrayList<>();
        vocabularyList = new ArrayList<>();
        vocabularyListMap = new HashMap<>();
        testDocumentClass = new ArrayList<>();
        testClassValues = new HashMap<>();
        testFileNames = new ArrayList<>();
        
        TRAINFOLDERNAME = args[0] + "//train";
        TESTFOLDERNAME = args[1];
        STOPWORDS = "StopWords.txt";
        CLASSNAMES = args[0]+ "//class_name.txt";
        DEV_LABEL = args[0] + "//dev_label.txt";
        OUTPUT_FILE = args[2];
        this.stopListEnable = stopListEnable;
        
        if (this.stopListEnable) {
            //Populate the stop words
            File stopWordsFile = new File(STOPWORDS);
            stopWords = readStopFile(stopWordsFile);
        } else {
            stopWords = null;
        }

        //Read the class_name.txt and populate classValueString
        readClassValues(CLASSNAMES);
    }

    public void populateVocabularyList(File folderName) throws FileNotFoundException {
        File folder = folderName;
        File[] listOfFiles = folder.listFiles();
        List<String> words;
        for (File spamFile : listOfFiles) {
            //READ FROM FILE
            words = readFile(spamFile);
            //COMPUTE THE VOCABULARY LIST
            for (String word : words) {
                if (word.equals("class")) {
                    continue;
                }
                if (!vocabularyListMap.containsKey(word)) {
                    vocabularyList.add(word);
                    vocabularyListMap.put(word, vocabularyList.size() - 1);
                }
            }
        }
    }

    public void populateWordCountTrainData(File folderName) throws FileNotFoundException {
        File folder = folderName;
        File[] listOfFiles = folder.listFiles();
        //Populate the wordCount for all the files
        for (File eachFile : listOfFiles) {
            //Read the file and COMPUTE THE X vector for Perceptron weight displacement calculation
            documentsAll.add(populateCountFromFile(eachFile));

            //Figure out the class of the document and save it in documentClass list
            if (classValueString.contains(folderName.getName())) {
                documentClass.add(Integer.valueOf(classValueString.indexOf(folderName.getName())));
            }
        }

        System.out.println(folderName + " Completed");
    }

    public void populateWordCountTestData(File testFolderName) throws FileNotFoundException {
        File testFolder = testFolderName;
        File[] listOfFiles = testFolder.listFiles();
        int classValue;
        int counter = 0;
        //Populate the wordCount for all the files
        for (File eachFile : listOfFiles) {
            //Read the file and COMPUTE THE X vector for Perceptron weight displacement calculation
            documentsAll.add(populateCountFromFile(eachFile));

            //Figure out the class of the test document and save it in testDocumentClass 
            classValue = testClassValues.get(eachFile.getName());
            testDocumentClass.add(classValue);
            counter++;
            if (counter % 200 == 0) {
                System.out.println((int) (counter / 20) + "% completed");
            }

        }

    }

    /**
     * Returns the .arff files for test data and train data
     *
     * @throws FileNotFoundException
     */
    public void generateARFFFilesTrain() throws FileNotFoundException, ParseException, IOException, Exception {

        //Populate the vocabulary list
        File folder = new File(TRAINFOLDERNAME);
        File[] listOfFolders = folder.listFiles();
        for (File trainFolder : listOfFolders) {
            populateVocabularyList(trainFolder);
        }

        System.out.println("Number of attributes: " + vocabularyList.size());

        //Populate the wordCount for Spam and Ham files
        for (File trainFolder : listOfFolders) {
            populateWordCountTrainData(trainFolder);
        }

        // 1. set up attributes
        List<Attribute> listOfAllAttributes = new ArrayList<>();
        Attribute attribute;
        for (int index = 0; index < vocabularyList.size(); index++) {
            attribute = new Attribute(vocabularyList.get(index));
            listOfAllAttributes.add(attribute);
        }
        // Declare the class attribute along with its values
        FastVector fvClassVal = new FastVector(classValueString.size());
        for (String classString : classValueString) {
            fvClassVal.addElement(classString);
        }
        Attribute ClassAttribute = new Attribute("class", fvClassVal);

        // Declare the feature vector
        FastVector allAttributes = new FastVector(vocabularyList.size() + 1);
        allAttributes.addAll(listOfAllAttributes);
        allAttributes.addElement(ClassAttribute);

        // 2. create Instances object
        Instances isTrainingSet = new Instances("train", allAttributes, documentsAll.size());
        // Set class index
        isTrainingSet.setClassIndex(allAttributes.size() - 1);
        Instance iExample;
        // 3. fill with data
        for (int counter = 0; counter < documentsAll.size(); counter++) {
            
            int[] indices = new int[documentsAll.get(counter).size()];
            int count=0;
            for(Object index : documentsAll.get(counter).keySet().toArray())
            {
                indices[count++] = (int)index;
            }
            
            Arrays.sort(indices);
            double[] attValues = new double[indices.length];
            
            for(int t=0; t<indices.length;t++)
            {
                attValues[t] = documentsAll.get(counter).get(indices[t]);
            }
            
            iExample = new SparseInstance(1, attValues, indices, vocabularyList.size());
            // Create the instance
            iExample.setValue(vocabularyList.size(), this.documentClass.get(counter));
            // add the instance
            isTrainingSet.add(iExample);
        }
        
          //output data into the arff file
//        ArffSaver saver = new ArffSaver();
//        saver.setInstances(isTrainingSet);
//        saver.setFile(new File("train.arff"));
//        saver.writeBatch();
//        System.out.println("TRAIN ARFF FILE GENERATED");


        
        documentsAll = new ArrayList<>();

        //Initialise the test mapping file for mapping documents to classes
        testClassValues = new HashMap<>();
        this.readTestDataClass(DEV_LABEL);

        //Populate the vocabulary list
        folder = new File(TESTFOLDERNAME);

        //Populate the wordCount
        populateWordCountTestData(folder);

        // 1. set up attributes
        listOfAllAttributes = new ArrayList<>();
        for (int index = 0; index < vocabularyList.size(); index++) {
            attribute = new Attribute(vocabularyList.get(index));
            listOfAllAttributes.add(attribute);
        }
        // Declare the class attribute along with its values
        //FastVector 
        fvClassVal = new FastVector(classValueString.size());
        
        for (String classString : classValueString) {
            fvClassVal.addElement(classString);
        }
       //Attribute 
        ClassAttribute = new Attribute("class", fvClassVal);

        // Declare the feature vector
        //FastVector 
        allAttributes = new FastVector(vocabularyList.size() + 1);
        allAttributes.addAll(listOfAllAttributes);
        allAttributes.addElement(ClassAttribute);

        // 2. create Instances object
        Instances isTestingSet = new Instances("test", allAttributes, documentsAll.size());
        // Set class index
        isTestingSet.setClassIndex(allAttributes.size() - 1);
        //Instance iExample;
        // 3. fill with data
        for (int counter = 0; counter < documentsAll.size(); counter++) {
            
            int[] indices = new int[documentsAll.get(counter).size()];
            int count=0;
            for(Object index : documentsAll.get(counter).keySet().toArray())
            {
                indices[count++] = (int)index;
            }
            
            Arrays.sort(indices);
            double[] attValues = new double[indices.length];
            
            for(int t=0; t<indices.length;t++)
            {
                attValues[t] = documentsAll.get(counter).get(indices[t]);
            }
            
            iExample = new SparseInstance(1, attValues, indices, vocabularyList.size());
            iExample.setValue(vocabularyList.size(), testDocumentClass.get(counter));

            // add the instance
            isTestingSet.add(iExample);
        }
        
/*******************ATTRIBUTE SELECTION USING INFO GAIN AND RANKER SELECTION*******************/ 
/****************************ENABLE THIS TO RUN LOGISTIC REGRESSION****************************/
//        AttributeSelection attributeSelection = new AttributeSelection();
//        String[] options = new String[4];
//        options[0] = "-E";
//        options[1] = "weka.attributeSelection.GainRatioAttributeEval";
//        options[2] = "-S";
//        options[3] = "weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N 10000";
//        attributeSelection.setOptions(options);
//        attributeSelection.setInputFormat(isTrainingSet);
//        Instances newTrain = Filter.useFilter(isTrainingSet, attributeSelection);  
//        Instances newTest = Filter.useFilter(isTestingSet, attributeSelection);
//        
//        System.out.println("Filered attributes train:" + newTrain.numAttributes());
//        System.out.println("Filered attributes test:" + newTest.numAttributes());
        
        
        /*******************************NAIVE BAYES MULTINOMIAL**********************/
        NaiveBayesMultinomial nb = new NaiveBayesMultinomial();
        nb.buildClassifier(isTrainingSet);
        Evaluation eval = new Evaluation(isTrainingSet);
        double[] result = eval.evaluateModel(nb,isTestingSet);
        //PRINT THIS RESULT INTO A FILE
        printOutputFile(result);
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
        
        
        
/******************LOGISTIC REGRESSION USING CONJUGATE GRAIDIENT DESCENT *************************/      
//        Logistic logisticClassifier = new Logistic();
//        logisticClassifier.setMaxIts(100);
//        logisticClassifier.setRidge(0.000001);
//        logisticClassifier.setUseConjugateGradientDescent(true);
//        try {
//            logisticClassifier.buildClassifier(newTrain);
//            Evaluation eval1 = new Evaluation(newTrain);
//            double[] result = eval1.evaluateModel(logisticClassifier,newTrain);
//              //PRINT THIS RESULT INTO A FILE
//            printOutputFile(result);
//            System.out.println(eval1.toSummaryString("\nTRAINING DATA RESULTS\n======\n", false));
//            eval1.evaluateModel(logisticClassifier,newTest);
//            System.out.println(eval1.toSummaryString("\nTESTING DATA RESULTS\n======\n", false));
//        } catch (Exception ex) {
//            System.out.println("Build Classifier for Logistic Regression failed.");
//        }
        
    }

    /**
     *
     * Calculate the wordCount from the file
     *
     * @param file
     * @return wordCount for the file
     * @throws FileNotFoundException
     */
    public HashMap<Integer, Integer> populateCountFromFile(File file) throws FileNotFoundException {
        int index;
        HashMap<Integer, Integer> wordCount = new HashMap<>();

        //Read all the words from the file
        List<String> words = readFile(file);

        //COMPUTE THE X vector for Perceptron weight displacement calculation
        for (String word : words) {

            if (word.equals("class")) {
                continue;
            }

            //ADD THE WORD TO THE VOCABULARY LIST
            if (vocabularyListMap.get(word) != null) {
                index = vocabularyListMap.get(word);
                if(!wordCount.containsKey(index))
                {
                    wordCount.put(index, 0);
                }
                wordCount.put(index, wordCount.get(index) + 1);
            }
        }
        return wordCount;
    }

    public void readTestDataClass(String fileName) throws FileNotFoundException {
        Scanner scanner = new Scanner(new File(fileName));
        String[] temp;
        while (scanner.hasNextLine()) {
            temp = scanner.nextLine().split("[ ]");
            testClassValues.put(temp[0], Integer.parseInt(temp[1]));
            testFileNames.add(temp[0]);
        }
    }

    private void printOutputFile(double[] result) throws FileNotFoundException, IOException {
        File fout = new File(OUTPUT_FILE);
	FileOutputStream fos = new FileOutputStream(fout);
        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fos));
        File[] listOfFiles = new File(TESTFOLDERNAME).listFiles();
        
        HashMap<String, Integer> resultMap = new HashMap<>();
        int counter=0;
        for(File eachFile : listOfFiles)
        {
            resultMap.put(eachFile.getName(), (int)result[counter]);
            counter++;
        }
       
        counter=0;
        for(String fileName: testFileNames)
        {
            bw.write(fileName+ " " + resultMap.get(fileName) );
            counter++;
            bw.newLine();
        }
	bw.close();
    }
    
    
    /**
     * Read the class_name.txt and populate classValueString
     *
     * @param fileName
     */
    public void readClassValues(String fileName) throws FileNotFoundException {
        Scanner scanner = new Scanner(new File(fileName));
        String[] temp;
        classValueString = new ArrayList<>();
        while (scanner.hasNextLine()) {
            temp = scanner.nextLine().split("[ ]");
            classValueString.add(temp[1]);
        }
        scanner.close();
        scanner = null;
    }

    /**
     * Reads from the stop file and returns the list of all words in it
     *
     * @param stopFile
     * @return
     * @throws FileNotFoundException
     */
    private List<String> readStopFile(File stopFile) throws FileNotFoundException {

        Scanner scanner = new Scanner(stopFile);
        List<String> words;
        words = new ArrayList<>();
        String temp;

        while (scanner.hasNextLine()) {
            temp = scanner.nextLine();
            words.add(temp.toLowerCase());
        }
        scanner.close();
        scanner = null;
        return words;
    }

    /**
     * Read from the file specified and returns a list of all the individual
     * words in the document.
     *
     * @param spamFile
     * @return list of all valid file in the file
     * @throws FileNotFoundException
     */
    public List<String> readFile(File file) throws FileNotFoundException {

        //Regex string [:'\\/@,.!?()*%+_{}<>0-9]
        Scanner scanner = new Scanner(file);
        List<String> words;
        words = new ArrayList<>();
        List<String> temp;
        List<String> tempStr;
        while (scanner.hasNextLine()) {
            temp = Arrays.asList(scanner.nextLine().split("[ :',.!?;]"));
            for (String t : temp) {
                t = t.toLowerCase();
                if (!t.matches("[a-z]+") || (stopListEnable && stopWords.contains(t)) || t.length() < 2) {
                    continue;
                } else {
                    words.add(t);
                }
            }
        }
        scanner.close();
        scanner = null;
        return words;
    }

    /**
     * @param args the command line arguments
     * @throws java.io.FileNotFoundException
     */
    public static void main(String[] args) throws FileNotFoundException, ParseException, IOException, Exception {

        //Perceptron without StopList
        NNWeka NeuralNetworkArffData;
        try {
            //removing stop words
            NeuralNetworkArffData = new NNWeka(args, true);
            NeuralNetworkArffData.generateARFFFilesTrain();
            //NeuralNetworkArffData.generateARFFFilesTest();

        } catch (FileNotFoundException ex) {
            Logger.getLogger(NNWeka.class.getName()).log(Level.SEVERE, null, ex);
            ex.printStackTrace();
        }
    }

}

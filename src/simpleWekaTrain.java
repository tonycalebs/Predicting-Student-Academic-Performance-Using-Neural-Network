
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import static java.lang.Math.random;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Debug.Random;
import weka.core.PropertyPath.Path;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author CALEB
 */
public class simpleWekaTrain {
   static  Instances train;
    static MultilayerPerceptron mlp;
    public static void simpleWekaTrain(String filepath) throws IOException{
        try{
            //Reading arff file
            mlp = new MultilayerPerceptron();
            FileReader trainreader = new FileReader(filepath);
            train = new Instances(trainreader);
            train.setClassIndex(train.numAttributes()-1);
            
            //setting parameters
            mlp.setLearningRate(0.1);
            mlp.setTrainingTime(500);
            mlp.setHiddenLayers("20");
            mlp.buildClassifier(train);
            
            
        }
        catch(Exception ex){
            ex.printStackTrace();
        }
        
          
    }
    public void NNEvalluation() throws Exception{
        try {
            Evaluation eval = new Evaluation(train);
            eval.evaluateModel(mlp, train);
            System.out.println(eval.errorRate());
            //Printing the mean root squared error
            System.out.println(eval.toSummaryString());//Summary of Training
          //  eval.crossValidateModel(mlp, train, 20, new Random(1));
        } catch (Exception ex) {
            Logger.getLogger(simpleWekaTrain.class.getName()).log(Level.SEVERE, null, ex);
        }
        saveModel();
        
    }
    
    public void saveModel() throws Exception{
        mlp.buildClassifier(train);
        weka.core.SerializationHelper.write("/generated.model", mlp);
        System.out.println("Model Generated");
        
    }
    
    public void readPredictmodel() throws Exception{
        MultilayerPerceptron mlp = (MultilayerPerceptron) weka.core.SerializationHelper.read("String Model Path");
        Instances datapredict = new Instances(new BufferedReader(new FileReader("Predictdatapath")));
        datapredict.setClassIndex(datapredict.numAttributes()-1);
        Instances predicteddata = new Instances(datapredict);
            for (int i = 0; i < datapredict.numInstances(); i++) {
            double clsLabel = mlp.classifyInstance(datapredict.instance(i));
            predicteddata.instance(i).setClassValue(clsLabel);
            }
            //Storing again in arff
            BufferedWriter writer = new BufferedWriter(
            new FileWriter("Output File Path"));
            writer.write(predicteddata.toString());
            writer.newLine();
            writer.flush();
            writer.close();


    }
    
    //classify unlabelled data
    
    public static void main (String [] args) throws IOException{
        simpleWekaTrain("calebdataset.arff");
    }
}

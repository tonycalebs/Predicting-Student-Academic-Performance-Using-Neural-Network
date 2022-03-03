import java.io.*; 
import java.util.logging.Level;
import java.util.logging.Logger;

//import static simpleWekaTrain.simpleWekaTrain(String).simpleWekaTrain(String).mlp;
import weka.core.*; 
import weka.core.Instances; 
import weka.classifiers.Evaluation; 
import weka.classifiers.*; 
import weka.classifiers.functions.MultilayerPerceptron;



public class training{ 
static MultilayerPerceptron mlp;
        training(){ 

                try{ 
FileReader trainreader = new FileReader("calebdataset.arff"); 
FileReader testreader = new FileReader("TESTDATA.arff"); 


Instances train = new Instances(trainreader); 
Instances test = new Instances(testreader); 
train.setClassIndex(train.numAttributes() - 1); 
test.setClassIndex(test.numAttributes() - 1); 

mlp = new MultilayerPerceptron();
mlp.setOptions(Utils.splitOptions("-L 0.4 -M 0.2 -N 500 -V 0 -S 0 -E 20 -H 20")); 


mlp.buildClassifier(train); 


Evaluation eval = new Evaluation(train); 
eval.evaluateModel(mlp, test); 
 mlp.buildClassifier(train);
        weka.core.SerializationHelper.write("generated.model", mlp);
        System.out.println("Model Generated");
System.out.println(eval.toSummaryString("\nResults\n======\n", false)); 
trainreader.close(); 
testreader.close(); 


} catch(Exception ex){ 

ex.printStackTrace(); 

} 

} 
        
        public static void Predictor(String path) throws Exception{
            try {
                Instances datapredict = new Instances(new BufferedReader(new FileReader(path)));
                datapredict.setClassIndex(datapredict.numAttributes()-1);
                Instances predicteddata = new Instances(datapredict);
                for (int i = 0; i < datapredict.numInstances(); i++) {
                    double clsLabel = mlp.classifyInstance(datapredict.instance(i));
                    predicteddata.instance(i).setClassValue(clsLabel);
                }
                //Storing again in arff
                System.out.println("");
                
                System.out.println(predicteddata.toString());
                BufferedWriter writer = new BufferedWriter(
                        new FileWriter("result.arff"));
                writer.write(predicteddata.toString());
                writer.newLine();
                writer.flush();
                writer.close();
            } catch (IOException ex) {
                Logger.getLogger(training.class.getName()).log(Level.SEVERE, null, ex);
            }
        }

public static void main(String args[]) throws Exception{ 

new training();
Predictor("predict.arff");

} 

} 

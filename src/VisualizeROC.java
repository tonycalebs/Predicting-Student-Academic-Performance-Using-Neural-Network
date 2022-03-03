import weka.classifiers.evaluation.ThresholdCurve;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.ThresholdVisualizePanel;

import java.awt.BorderLayout;

import javax.swing.JFrame;
public class VisualizeROC {
  
 
  public static void main(String[] args) throws Exception {
    Instances curve = DataSource.read("result.arff");
    curve.setClassIndex(curve.numAttributes() - 1);
    
    // method visualize
    ThresholdVisualizePanel tvp = new ThresholdVisualizePanel();
    tvp.setROCString("(Area under ROC = " + 
        Utils.doubleToString(ThresholdCurve.getROCArea(curve), 4) + ")");
    tvp.setName(curve.relationName());
    PlotData2D plotdata = new PlotData2D(curve);
    plotdata.setPlotName(curve.relationName());
    plotdata.addInstanceNumberAttribute();
    // specify which points are connected
    boolean[] cp = new boolean[curve.numInstances()];
    for (int n = 1; n < cp.length; n++)
      cp[n] = true;
    plotdata.setConnectPoints(cp);
    // add plot
    tvp.addPlot(plotdata);
    
    // method visualizeClassifierErrors
    final JFrame jf = new JFrame("WEKA ROC: " + tvp.getName());
    jf.setSize(500,400);
    jf.getContentPane().setLayout(new BorderLayout());
    jf.getContentPane().add(tvp, BorderLayout.CENTER);
    jf.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
    jf.setVisible(true);
  }
}

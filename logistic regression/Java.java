import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class TableModel {
    double learningRate = 0.01;
    int numIter = 1000;
    List<Double> theta;

    public TableModel() {

    }

    public TableModel(List<Double> x) {
        this.theta = x;
    }

    public TableModel(double lr, int num) {
        this.learningRate = lr;
        this.numIter = num;
    }


    public List<List<Double>> prepareX(List<List<Double>> x) {
        for (List<Double> row : x) {
            row.add(0, 1.0);
        }
        return x;
    }


    public double logistic(double z) {
        return 1/(1 + Math.exp(z));
    }

    public double dotProduct(List<Double> x, List<Double> y) {
        if (x.size() != y.size()) {
            return Double.NaN;
        }
        double ret = 0;
        for (int i = 0; i < x.size(); i++) {
            ret += x.get(i) * y.get(i);
        }
        return ret;
    }

    public List<Double> subtract(List<Double> x, List<Double> y) {
        if (x.size() != y.size()) {
            return null;
        }

        List<Double> s = new ArrayList<>(x.size());
        for (int i = 0; i < x.size(); i++) {
            s.add(x.get(i) - y.get(i));
        }
        return s;
    }

    public List<Double> multiply(List<Double> x, double y) {
        List<Double> result = new ArrayList<>();
        for (double doubles: x) {
            result.add(doubles * y);
        }
        return result;
    }

    public void fit(List<List<Double>> x, List<Double> y) {
        List<List<Double>> xNew = prepareX(x);
        theta = new ArrayList<Double>(Collections.<Double>nCopies(xNew.get(0).size(), 0.0));

        for (int i = 0; i < numIter; i++) {
            List<Double> z = new ArrayList<>();

            for (List<Double> doubleList : xNew) {
                double sum = 0;
                sum = dotProduct(theta, doubleList);
                z.add(sum);
            }

            List<Double> h = new ArrayList<>(z.size());

            for (Double aDouble : z) {
                h.add(logistic(aDouble));
            }

            List<List<Double>> transposed = new ArrayList<>();
            for (int j = 0; j < xNew.get(0).size(); j++) {
                List<Double> row = new ArrayList<>();
                for (List<Double> doubles : xNew) {
                    row.add(doubles.get(j));
                }
                transposed.add(row);
            }

            List<Double> gradients = new ArrayList<>();
            for (List<Double> doubles : transposed) {
                double grad = 0;
                grad = dotProduct(subtract(h, y), doubles) / xNew.size();
                gradients.add(grad);
            }

            theta = subtract(theta, multiply(gradients, learningRate));

        }
    }

    public double predict(List<Double> x) {
        return logistic(dotProduct(x, theta));
    }

    public void saveModel(String path) throws IOException {
        ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(path));
        out.writeObject(theta);
        out.flush();
        out.close();
        System.out.println("model saved");
    }
}

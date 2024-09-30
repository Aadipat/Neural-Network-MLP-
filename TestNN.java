import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import matrix.Matrix;
import neuralnetwork.NeuralNetwork;

class testnn {
    public testnn() {
    }

    public static void main(String[] var0) {
        int[] var1 = new int[] { 2, 2, 1 };
        NeuralNetwork var2 = new NeuralNetwork(var1);
        List<Matrix> var3 = new ArrayList<>();
        List<Matrix> var4 = new ArrayList<>();
        Matrix var5 = new Matrix(new double[][] { { 1.0, 1.0 } }, 1, 2);
        Matrix var6 = new Matrix(new double[][] { { 1.0, 0.0 } }, 1, 2);
        Matrix var7 = new Matrix(new double[][] { { 0.0, 1.0 } }, 1, 2);
        Matrix var8 = new Matrix(new double[][] { { 0.0, 0.0 } }, 1, 2);
        var3.add(var5);
        var3.add(var6);
        var3.add(var7);
        var3.add(var8);
        Matrix var9 = new Matrix(new double[][] { { 0.0 } }, 1, 1);
        Matrix var10 = new Matrix(new double[][] { { 1.0 } }, 1, 1);
        Matrix var11 = new Matrix(new double[][] { { 1.0 } }, 1, 1);
        Matrix var12 = new Matrix(new double[][] { { 1.0 } }, 1, 1);
        var4.add(var9);
        var4.add(var10);
        var4.add(var11);
        var4.add(var12);
        System.out.println(var2.feedforward(var3.get(0)));
        System.out.println(var2.feedforward(var3.get(1)));
        System.out.println(var2.feedforward(var3.get(2)));
        System.out.println(var2.feedforward(var3.get(3)));
        var2 = NeuralNetwork.find_best_trained_model(1000, var3, var4, 1000, 0.01, var1, 1);
        System.out.println(var2.feedforward(var3.get(0)));
        System.out.println(var2.feedforward(var3.get(1)));
        System.out.println(var2.feedforward(var3.get(2)));
        System.out.println(var2.feedforward(var3.get(3)));
    }
}

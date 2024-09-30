package neuralnetwork;

import matrix.*;

import java.io.*;
import java.util.*;
import java.util.function.Function;

public class NeuralNetwork {

    // MODEL weights and biases
    Matrix[] weights;
    Matrix[] biases;

    // EXPLICIT Activation function and its derivative
    Function<Double, Double> activationFunction =
            (x) -> 1 / (1 + Math.exp(-x));
    Function<Double, Double> derivateActivationFunction =
            (x) -> (Math.exp(-x)) / (Math.pow((1 + Math.exp(-x)), 2));

    // Array of all intermediate outputs
    private final Matrix[] all_outputs;

    // Dimensions of Artificial Neural Network, default a perceptron
    int[] nodes_array;

    public NeuralNetwork(int[] nodes_array) {
        this.weights = new Matrix[nodes_array.length - 1];
        this.biases = new Matrix[nodes_array.length - 1];
        this.nodes_array = nodes_array;

        for (int i = 0; i < this.weights.length; i++) {
            Matrix m = new Matrix(nodes_array[i], nodes_array[i + 1]);
            m.randomise();
            this.weights[i] = m;
        }
        for (int i = 0; i < this.biases.length; i++) {
            Matrix m = new Matrix(1, nodes_array[i + 1]);
            m.randomise();
            this.biases[i] = m;
        }
        this.all_outputs = new Matrix[nodes_array.length];
    }

    public void setActivation(Function<Double, Double> A, Function<Double, Double> derivativeA){
        this.activationFunction = A;
        this.derivateActivationFunction = derivativeA;
    }

    public Matrix feedforward(Matrix input) {
        this.all_outputs[0] = input;
        for (int i = 0; i < this.weights.length; i++) {
            input = Matrix.multiply(input, this.weights[i]);
            input.add(this.biases[i]);
            input.apply_function(activationFunction);
            this.all_outputs[i + 1] = input;
        }
        return input;
    }

    public void gradient_descent_One_Level(int layer, Matrix dEBydO_layer, double learning_rate) {

        Matrix previous_layer_output = this.all_outputs[layer - 1].copy();

        Matrix input_to_layer_without_activation = Matrix.multiply(previous_layer_output, weights[layer - 1]);

        input_to_layer_without_activation.apply_function(derivateActivationFunction);

        Matrix deltaB = Matrix.hadamard_product(dEBydO_layer,
                input_to_layer_without_activation);

        deltaB.apply_scalar(learning_rate);

        previous_layer_output.Transpose();

        Matrix deltaW = Matrix.multiply(previous_layer_output, deltaB);

        this.weights[layer - 1] = Matrix.subtract(this.weights[layer - 1], deltaW);
        this.biases[layer - 1] = Matrix.subtract(this.biases[layer - 1], deltaB);
    }

    public void learn(double h, Matrix dEBydOl) {
        // Applying gradient descend on the weights and biases.
        // E is error: E = (1/2)(Sum (O-T)^2)
        // dE/dO = sum(O-T)
        // To do this we need the Gradient E vector with respect to the weights and
        // biases. => delta W = -h*gradient E with respect to W
        // => delta B = -h*gradient E with respect to B

        Matrix error_derivative_Previous = dEBydOl.copy();

        Matrix error_derivative_layer = dEBydOl.copy();

        for (int i = this.nodes_array.length - 1; i >= 1; i--) {

            // Gradient descent for output layer:
            if (i == this.nodes_array.length - 1) {
                error_derivative_layer = dEBydOl.copy();
            } else {
                // Applying BACKPROPAGATION ALGORITHM to find
                // error derivative with respect to hidden layers.

                Matrix input_to_layer_without_activation = Matrix.multiply(this.all_outputs[i], weights[i]);

                input_to_layer_without_activation.apply_function(derivateActivationFunction);

                Matrix dATimesDlast = Matrix.hadamard_product(error_derivative_Previous,
                        input_to_layer_without_activation);
                Matrix weights = this.weights[i].copy();

                weights.Transpose();
                error_derivative_layer = Matrix.multiply(dATimesDlast, weights);
            }

            gradient_descent_One_Level(i, error_derivative_layer, h);
            error_derivative_Previous = error_derivative_layer;
        }

    }

    public Matrix getCost(List<Matrix> inputs, List<Matrix> targets, boolean sumSquares) {
        Matrix error = Matrix.zeroes(targets.getFirst().rows, targets.getFirst().cols);
        for (int i = 0; i < inputs.size(); i++) {
            Matrix output = this.feedforward(inputs.get(i));
            Matrix diff = Matrix.subtract(output, targets.get(i));
            if(sumSquares){
                error.add(Matrix.hadamard_product(diff,diff));
            } else {
                error.add(diff);
            }
        }
        return error;
    }


    public void trainModel(List<Matrix> inputs, List<Matrix> targets,
                      int epochs, double learning_rate,
                      int learnPerEpoch, boolean printEpochCost){
        for(int e = 0;e<epochs;e++){

            Matrix sum_differences = Matrix.zeroes(targets.getFirst().rows, targets.getFirst().cols);
            // E = (Sum of O-T squared divided by 2)
            // Sum of error over the epoch
            Matrix cost = Matrix.zeroes(targets.getFirst().rows, targets.getFirst().cols);

            for (int i = 0; i < inputs.size(); i++) {
                Matrix output = feedforward(inputs.get(i));
                Matrix diff = Matrix.subtract(output, targets.get(i));
                sum_differences.add(diff);
                Matrix diffSquared = Matrix.hadamard_product(diff, diff);

                if((i + 1) % (inputs.size()/learnPerEpoch) == 0) {
                    learn(learning_rate, sum_differences);
                    sum_differences = Matrix.zeroes(targets.getFirst().rows, targets.getFirst().cols);
                }

                cost.add(diffSquared);
            }

            if (printEpochCost) {
                System.out.print("Epoch " + e + " : Cost is ");
                System.out.println(getCost(inputs, targets, true));
            }
        }

    }


    public static NeuralNetwork find_best_trained_model(int number_of_random_models,
                                                        List<Matrix> inputs, List<Matrix> targets,
                                                        int epochs, double learning_rate,
                                                        int[] nodes_array, int learnPerEpoch) {

        NeuralNetwork[] nns = new NeuralNetwork[number_of_random_models];

        for (int i = 0; i < nns.length; i++) {
            nns[i] = new NeuralNetwork(nodes_array);
        }

        NeuralNetwork nn;

        Matrix[] costs = new Matrix[nns.length];

        for (int i = 0; i < nns.length; i++) {
            nns[i].trainModel(inputs, targets, epochs, learning_rate, learnPerEpoch, false);
            Matrix cost = nns[i].getCost(inputs, targets, true);
            costs[i] = cost;
            System.out.print("Random model " + i + ": ");
            System.out.println(cost);
        }

        int best = 0;
        Matrix best_m = costs[best];

        for (int i = 1; i < costs.length; i++) {
            if (best_m.is_larger_than(costs[i])) {
                best = i;
                best_m = costs[i];
            }
        }

        System.out.println("Best model is of index: " + best);
        System.out.print("It has cost ");
        System.out.println(best_m);

        nn = nns[best];

        return nn;
    }

    public static NeuralNetwork combine_models(NeuralNetwork[] models) {
        NeuralNetwork final_model;

        int[] nodes_array = new int[0];
        int previous_network_output = models[0].nodes_array[models[0].nodes_array.length - 1];
        int layers = models[0].nodes_array.length;

        for (int i = 1; i < models.length; i++) {
            if (models[i].nodes_array[0] != previous_network_output) {
                System.out.println(models[i].nodes_array[0] + " " + previous_network_output);
                System.out.println("Not cohesive network, cannot combine");
                return new NeuralNetwork(null);
            }
            layers += models[i].nodes_array.length - 1;
        }

        nodes_array = new int[layers];
        int k = 0;
        for (int n = 0; n < models.length; n++) {
            for (int i = 1; i < models[n].nodes_array.length; i++) {
                nodes_array[k] = models[n].nodes_array[i];
                k++;
            }
        }

        final_model = new NeuralNetwork(nodes_array);
        int c = 0;
        for (int n = 0; n < models.length; n++) {
            for (int i = 0; i < models[n].weights.length; i++) {
                final_model.weights[c] = models[n].weights[i];
                final_model.biases[c] = models[n].biases[i];
                c++;
            }
        }

        return final_model;
    }

    public void save_model(String filename, String filePath) {
        try {
            File myFile = new File(filePath + filename);
            if (myFile.createNewFile()) {
                System.out.println("File created: " + myFile.getName());
            } else {
                System.out.println("File already exists.");
            }
        } catch (IOException e) {
            System.out.println("An error occurred.");
        }

        try {
            FileWriter myWriter = new FileWriter(filePath + filename);

            StringBuilder model_as_string = new StringBuilder();

            model_as_string.append("{");

            for (Matrix w : this.weights) {
                model_as_string.append(w.toString());
            }

            model_as_string.append("}\n{");

            for (Matrix b : this.biases) {
                model_as_string.append(b.toString());
            }

            model_as_string.append("}");

            myWriter.write(model_as_string.toString());
            myWriter.close();
            System.out.println("Successfully wrote to the file.");
        } catch (IOException e) {
            System.out.println("An error occurred.");
        }

    }

    public static NeuralNetwork LoadModel(String filename) {
        NeuralNetwork nn;

        Matrix[] weights = new Matrix[0];
        Matrix[] biases = new Matrix[0];
        try {
            File myFile = new File("../target/saved_models/" + filename);
            Scanner myReader = new Scanner(myFile);

            // go through file and retrieve data.
            // In doing so find the length of network.
            StringBuilder model = new StringBuilder();
            int count = 0;
            boolean length_found = false;
            while (myReader.hasNextLine()) {
                String data = myReader.nextLine();
                if (!length_found) {
                    count++;
                }
                char[] chars = data.toCharArray();
                // As each matrix is on one line.
                for (char c : chars) {
                    if (c == '}' && !length_found) {
                        length_found = true;

                        break;
                    }
                }
                model.append(data).append('\n');
            }
            // System.out.print(model);

            // Adjust the weights and biases arrays with the correct lengths
            weights = new Matrix[count - 1];
            biases = new Matrix[count - 1];

            // Go through model string to now initialise the individual matrices.
            int bi = 0;
            int wi = 0;
            char[] chars = model.toString().toCharArray();
            // As each matrix is on one line.
            boolean weights_done = false;
            StringBuilder line = new StringBuilder();
            for (int c = 0; c < chars.length; c++) {
                if (chars[c] == '}') {
                    weights_done = true;
                    c = c + 2;
                    continue;
                }
                if (chars[c] != '{' && chars[c] != '}') {
                    line.append(chars[c]);
                }
                if (chars[c] == '\n') {
                    Matrix m = Matrix.read_Matrix(line.toString());
                    if (weights_done) {
                        biases[bi] = m;
                        bi++;
                    } else {
                        weights[wi] = m;
                        wi++;
                    }

                    line = new StringBuilder();
                }
            }

            myReader.close();

        } catch (FileNotFoundException e) {
            System.out.println("An error occurred...");
        }

        // Find the nodes and initialise the network correctly

        int[] nodes_array = new int[biases.length + 1];

        nodes_array[0] = weights[0].rows;

        for (int b = 1; b < nodes_array.length; b++) {
            nodes_array[b] = biases[b - 1].cols;
        }

        nn = new NeuralNetwork(nodes_array);
        nn.weights = weights;
        nn.biases = biases;

        return nn;
    }

}

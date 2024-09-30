package matrix;

import java.util.Arrays;
import java.util.function.Function;

public class Matrix {
    public double[][] matrix;
    public int cols;
    public int rows;

    public Matrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.matrix = new double[rows][cols];
        for (int i = 0; i < this.matrix.length; i++) {
            for (int j = 0; j < this.matrix[i].length; j++) {
                if (i == j) {
                    this.matrix[i][j] = 1;
                } else {
                    this.matrix[i][j] = 0;
                }
            }
        }
    }

    public static Matrix zeroes(int rows, int cols) {
        Matrix m = new Matrix(rows, cols);
        for (int i = 0; i < m.matrix.length; i++) {
            Arrays.fill(m.matrix[i], 0);
        }
        return m;
    }

    public Matrix(double[][] vals, int rows, int cols) {
        this.matrix = vals;
        this.rows = rows;
        this.cols = cols;
    }

    public void randomise() {
        for (int i = 0; i < this.matrix.length; i++) {
            for (int j = 0; j < this.matrix[i].length; j++) {
                double num = Math.random();
                if (num < 0.5) {
                    this.matrix[i][j] = -Math.random();
                } else {
                    this.matrix[i][j] = Math.random();
                }
            }
        }
    }

    public void add(Matrix m) {

        if (this.rows == m.rows && this.cols == m.cols) {
            for (int i = 0; i < this.matrix.length; i++) {
                for (int j = 0; j < this.matrix[i].length; j++) {
                    this.matrix[i][j] += m.matrix[i][j];
                }
            }
        } else {
            System.out.println(" Matrices are of different sizes!!");
        }

    }

    public static Matrix subtract(Matrix m1, Matrix m2) {
        Matrix m3 = new Matrix(m1.rows, m2.cols);
        if (m1.rows == m2.rows && m1.cols == m2.cols) {
            for (int i = 0; i < m1.matrix.length; i++) {
                for (int j = 0; j < m2.matrix[i].length; j++) {
                    m3.matrix[i][j] = m1.matrix[i][j] - m2.matrix[i][j];
                }
            }
            return m3;
        } else {
            System.out.println(" Matrices are of different sizes!! Cannot Subtract");
            System.out.println(m1);
            System.out.println(m2);
        }
        return m3;
    }

    public void Transpose() {
        Matrix m = new Matrix(this.cols, this.rows);

        for (int i = 0; i < this.cols; i++) {
            for (int j = 0; j < this.rows; j++) {
                m.matrix[i][j] = this.matrix[j][i];
            }
        }
        int temp = this.rows;
        this.rows = this.cols;
        this.cols = temp;
        this.matrix = m.matrix;
    }

    public void apply_scalar(double scalar) {
        for (int i = 0; i < this.matrix.length; i++) {
            for (int j = 0; j < this.matrix[i].length; j++) {
                this.matrix[i][j] *= scalar;
            }
        }
    }

    public void apply_function(Function<Double, Double> function) {
        for (int i = 0; i < this.matrix.length; i++) {
            for (int j = 0; j < this.matrix[i].length; j++) {
                this.matrix[i][j] = function.apply(this.matrix[i][j]);
            }
        }
    }

    public static Matrix multiply(Matrix m1, Matrix m2) {
        Matrix m3 = new Matrix(null, m1.rows, m2.cols);
        if (m1.cols == m2.rows) {

            m3 = new Matrix(m1.rows, m2.cols);

            for (int i = 0; i < m3.rows; i++) {
                for (int j = 0; j < m3.cols; j++) {
                    double product_sum = 0;
                    for (int k = 0; k < m2.rows; k++) {
                        product_sum += m1.matrix[i][k] * m2.matrix[k][j];
                    }
                    m3.matrix[i][j] = product_sum;
                }
            }

        } else {
            System.out.println("Matrices cannot be multiplied!!!");
            System.out.println(m1);
            System.out.println(m2);
        }

        return m3;
    }

    public static Matrix hadamard_product(Matrix m1, Matrix m2) {
        Matrix m3 = new Matrix(null, m1.rows, m2.cols);

        if (m1.rows == m2.rows && m1.cols == m2.cols) {
            m3 = new Matrix(m1.rows, m2.cols);
            for (int i = 0; i < m3.rows; i++) {
                for (int j = 0; j < m3.cols; j++) {
                    m3.matrix[i][j] = m1.matrix[i][j] * m2.matrix[i][j];
                }
            }
            return m3;
        }

        return m3;
    }

    public boolean is_larger_than(Matrix m) {

        Matrix difference = Matrix.subtract(this, m);
        for (double[] r : difference.matrix) {
            for (double d : r) {
                if (d < 0) {
                    return false;
                }
            }
        }
        return true;
    }

    public Matrix copy() {
        double[][] copy = new double[rows][cols];
        for(int i = 0;i<rows;i++){
            System.arraycopy(matrix[i], 0, copy[i], 0, cols);
        }
        return new Matrix(copy, this.rows, this.cols);
    }

    public static Matrix read_Matrix(String line) {
        char[] chars = line.toCharArray();
        int cols = 0;
        int rows = 0;
        boolean row_changed = false;
        for (int c = 2; c < chars.length - 2; c++) {
            if (chars[c] == ',' && !row_changed) {
                cols++;
            }
            if (chars[c] == ']') {
                rows++;
                row_changed = true;
            }

        }
        double[][] matrix = new double[rows][cols + 1];

        int r = 0;
        int co = 0;

        boolean num_start = false;
        boolean num_end = false;
        StringBuilder temp = new StringBuilder();
        for (int c = 2; c < chars.length - 2; c++) {
            if (chars[c] == ']') {
                num_end = true;
            } else if (chars[c] == '[') {
                continue;
            } else if (chars[c] == ' ') {
                num_start = true;
            } else if (chars[c] == ',') {
                num_end = true;
            } else {
                temp.append(chars[c]);
            }

            if (num_end && num_start) {
                num_end = false;
                num_start = false;
                double num = Double.parseDouble(temp.toString());
                if (co > cols) {
                    co = 0;
                    r++;
                }
                matrix[r][co] = num;
                co++;
                temp = new StringBuilder();
            }

        }

        return new Matrix(matrix, rows, cols + 1);
    }

    @Override
    public String toString() {
        StringBuilder str = new StringBuilder();

        str.append("[");
        for (double[] doubles : this.matrix) {
            str.append("[");

            for (int j = 0; j < doubles.length - 1; j++) {
                str.append(" ").append(doubles[j]).append(",");

            }

            str.append(" ").append(doubles[doubles.length - 1]);
            str.append("]");
        }

        str.append("]\n");

        return str.toString();
    }
}

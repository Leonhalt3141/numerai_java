package models;

import data.*;
import org.apache.avro.generic.*;
import org.apache.parquet.hadoop.*;
import smile.data.DataFrame;
import smile.math.*;
import smile.regression.*;
import smile.validation.*;
import smile.data.formula.*;
import smile.data.type.*;
import smile.data.formula.Terms.*;
import ml.dmlc.xgboost4j.java.Booster;

import ml.dmlc.xgboost4j.java.XGBoost;

import java.io.*;
import java.util.*;
import java.util.stream.*;

public class Example {

    static int dataSize = 2420521;
    static int sampleSize = 10000;

    static int cvSize = 5;

    static String targetVariable = "normed_target";

    static String trainFile = "";

    static ReadData readData = new ReadData();

    public static void main(String[] args) {
        try {
            RegressionValidations<GradientTreeBoost> trainedRegressors = train();

            System.out.println(trainedRegressors);

        } catch (IOException e) {
            System.err.println(e.getMessage());
        }

    }

    public static int[] slice(int[] array, int startIndex, int endIndex) {
        return Arrays.copyOfRange(array, startIndex, endIndex);
    }

    public static double[] normalize(double value, double mean, double std) {
        return new double[]{(value - mean) / std};
    }

    public static RegressionValidations<GradientTreeBoost> train() throws IOException {
        MathEx.setSeed(19650218);

        int[] permutation = MathEx.permutate(dataSize);

        List<Integer> samplePermutation = Arrays.stream(slice(permutation, 0, sampleSize))
                .boxed().collect(Collectors.toList());

        int count = 0;
//        GenericRecord record;
        List<GenericRecord> recordList = new ArrayList<>();
        ParquetReader<GenericRecord> iter = readData.readAsIterator(trainFile);

        while (count < dataSize) {
            GenericRecord row = iter.read();
            if (samplePermutation.contains(count)) {
                recordList.add(0, row);
            }

            count += 1;
        }

        DataFrame trainData = readData.convertToDataFrame(recordList);

        double[] targetArray = trainData.column("target").toDoubleArray();

        double mean = MathEx.mean(targetArray);
        double std = MathEx.mean(targetArray);

        List<String> featureColumns = Arrays.stream(trainData
                .schema()
                .fields())
                .filter(x1 -> x1.name.contains("feature"))
                .map(x2 -> x2.name)
                .collect(Collectors.toList())
                .subList(0, 1205);

        featureColumns.add(targetVariable);

//        String[] featureColumnArray = new String[featureColumns.size()];
//        featureColumnArray = featureColumns.toArray(featureColumnArray);


        DataFrame selected = trainData
                .merge(
                        DataFrame.of(
                                Arrays.stream(trainData.column("target").toDoubleArray())
                                        .boxed()
                                        .map(value -> normalize(value, mean, std))
                                        .toArray(double[][]::new),
                                targetVariable
                        )
                )
                .select(featureColumns.toArray(new String[0]));

        Formula formula = Formula.lhs(targetVariable);
        Properties params = new Properties();
        params.put("nztree", 2000);
        params.put("maxDepth", 20);
        params.put("maxNodes", 15);
        params.put("nodeSize", 10);
        params.put("shrinkage", 0.01);
        params.put("subsample", 0.7);

        return CrossValidation.regression(
                cvSize,
                formula,
                selected,
                (f, d) -> GradientTreeBoost.fit(f, d, params)
        );
    }

}

package models;

import data.*;
import org.apache.avro.generic.*;
import org.apache.parquet.hadoop.*;
import smile.data.DataFrame;
import smile.data.Tuple;
import smile.math.*;
import smile.regression.*;
import smile.validation.*;
import smile.data.formula.*;
import smile.io.Write;
import smile.io.Read;

import smile.data.type.*;
import smile.data.formula.Terms.*;
import ml.dmlc.xgboost4j.java.Booster;

import ml.dmlc.xgboost4j.java.XGBoost;

import scala.collection.Iterator;

import java.io.*;
import java.util.*;
import java.util.stream.*;
import java.io.*;
import java.nio.file.Path;
import java.nio.file.Paths;


public class Example {

    static int dataSize = 2420521;
    static int sampleSize = 10000;

    static int cvSize = 5;

    static String targetVariable = "normed_target";

    static String trainFile = "data/494_v4_1_train.parquet";
    static String liveFile = "data/494_v4_1_live.parquet";

    static double targetMean;
    static double targetStd;

    static ReadData readData = new ReadData();

    public static void main(String[] args) {
        try {
            System.out.println("Start training");
            RegressionValidations<GradientTreeBoost> trainedRegressors = train();
//
            writeModel(trainedRegressors, "GBT_regressor.model");
//            RegressionValidations<GradientTreeBoost> trainedRegressors = readModel("GBT_regressor.model");
            System.out.println(trainedRegressors);

            DataFrame liveDataFrame = readDataFrame(liveFile);

            List<String> featureColumns = Arrays.stream(liveDataFrame
                            .schema()
                            .fields())
                    .filter(x1 -> x1.name.contains("feature"))
                    .map(x2 -> x2.name)
                    .collect(Collectors.toList())
                    .subList(0, 1205);

            featureColumns.add(targetVariable);

            DataFrame liveDF = liveDataFrame
                    .merge(
                            DataFrame.of(
                                    Arrays.stream(liveDataFrame.column("target").toDoubleArray())
                                            .boxed()
                                            .map(value -> normalize(value, targetMean, targetStd))
                                            .toArray(double[][]::new),
                                    targetVariable
                            )
                    )
                    .select(featureColumns.toArray(new String[0]));

            double[] predicted = predict(trainedRegressors, liveDF);

            double[][] data = Arrays.stream(predicted)
                    .boxed()
                    .map(value -> new double[]{(value * targetStd) + targetMean})
                    .toArray(double[][]::new);

            DataFrame predictedDF = liveDataFrame
                    .select("id")
                    .merge(
                            DataFrame.of(data, "prediction")
                    );

            writeLiveCSV(predictedDF, "live_result.csv");

        } catch (Exception e) {
            System.out.println("Failed.");
            System.err.println(e.getMessage());
        }

    }

    public static void writeModel(RegressionValidations<GradientTreeBoost> regressors, String modelPath) throws IOException {
        Write.object(regressors, Paths.get(modelPath));
    }

    public static RegressionValidations<GradientTreeBoost> readModel(String modelPath) throws Exception {

        return (RegressionValidations<GradientTreeBoost>) Read.object(Paths.get(modelPath));
    }

    public static DataFrame readDataFrame(String parquetPath) throws IOException {
        ParquetReader<GenericRecord> reader = readData.readAsIterator(parquetPath);
        List<GenericRecord> recordList = new ArrayList<>();

        GenericRecord row = reader.read();

        while (row != null) {
            recordList.add(row);
            row = reader.read();
        }

        return readData.convertToDataFrame(recordList);

    }

    public static void writeLiveCSV(DataFrame predictedDF, String csvPath) throws IOException {
        FileWriter csvWriter = new FileWriter(csvPath);

        int dfSize = predictedDF.size();

        for (int i = 0; i < dfSize; i++) {
            Tuple row = predictedDF.get(i);
            csvWriter
                    .append(row.get(0).toString())
                    .append(",")
                    .append(row.get(1).toString())
                    .append("\n");
        }

        csvWriter.flush();
        csvWriter.close();
    }

    public static double[] predict(RegressionValidations<GradientTreeBoost> trainedRegressors, DataFrame dataFrame) {
        int size = dataFrame.size();
        double[] results = new double[size];

        for (RegressionValidation<GradientTreeBoost> regressor : trainedRegressors.rounds) {
            double[] prediction = regressor.model.predict(dataFrame);
            
            for (int i = 0; i < size; i++) {
                results[i] = results[i] + prediction[i];
            }
        }

        for (int i = 0; i < size; i++) {
            results[i] = results[i] / (double) cvSize;
        }

        return results;
    }


    public static int[] slice(int[] array, int startIndex, int endIndex) {
        return Arrays.copyOfRange(array, startIndex, endIndex);
    }

    public static double[] normalize(double value, double mean, double std) {
        return new double[]{(value - mean) / std};
    }

    public static RegressionValidations<GradientTreeBoost> train() throws IOException {
        MathEx.setSeed(19650218);
        System.out.println("Load data");

        int[] permutation = MathEx.permutate(dataSize);

        List<Integer> samplePermutation = Arrays.stream(slice(permutation, 0, sampleSize))
                .boxed().collect(Collectors.toList());

        int count = 0;
        List<GenericRecord> recordList = new ArrayList<>();
        ParquetReader<GenericRecord> iter = readData.readAsIterator(trainFile);

        while (count < dataSize) {
            GenericRecord row = iter.read();
            if (samplePermutation.contains(count)) {
                recordList.add(0, row);
            }

            count += 1;
        }
        System.out.println("recordList: " + recordList.size());

        DataFrame trainData = readData.convertToDataFrame(recordList);

        double[] targetArray = trainData.column("target").toDoubleArray();

        targetMean = MathEx.mean(targetArray);
        targetStd = MathEx.sd(targetArray);

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
                                        .map(value -> normalize(value, targetMean, targetStd))
                                        .toArray(double[][]::new),
                                targetVariable
                        )
                )
                .select(featureColumns.toArray(new String[0]));

        Formula formula = Formula.lhs(targetVariable);
        Properties params = new Properties();
        params.put("smile.gradient_boost.trees", 500);
        params.put("smile.gradient_boost.max_depth", 20);
        params.put("smile.gradient_boost.max_nodes", 15);
        params.put("smile.gradient_boost.node_size", 10);
        params.put("smile.gradient_boost.shrinkage", 0.01);
        params.put("smile.gradient_boost.sampling_rate", 0.7);

        System.out.println("Train model");

        return CrossValidation.regression(
                cvSize,
                formula,
                selected,
                (f, d) -> GradientTreeBoost.fit(f, d, params)
        );
    }

}

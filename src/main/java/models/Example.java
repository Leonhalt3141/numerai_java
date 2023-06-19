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

import java.io.*;
import java.util.*;
import java.util.stream.*;
import java.nio.file.Paths;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Example {

    static int dataSize = 2420521;
    static int sampleSize = 10000;

    static int cvSize = 5;

    static String targetVariable = "normed_target";

    static String trainFile = "data/489_v4_1_train.parquet";
    static String liveFile = "data/493_v4_1_live.parquet";

    static ReadData readData = new ReadData();

    public static void main(String[] args) {
        Logger logger = LoggerFactory.getLogger(Example.class);
        try {
            logger.info("Start training");
            RegressionValidations<GradientTreeBoost> trainedRegressors = train();

            writeModel(trainedRegressors, "GBT_regressor.model");

            logger.info(trainedRegressors.toString());

            DataFrame liveDataFrame = readDataFrame(liveFile);

            double[] predicted = predict(trainedRegressors, liveDataFrame);

            double[][] data = Arrays.stream(predicted)
                    .boxed()
                    .toArray(double[][]::new);

            DataFrame predictredDF = liveDataFrame
                    .select("id")
                    .merge(
                            DataFrame.of(data, "prediction")
                    );

            writeLiveCSV(predictredDF, "live_result.csv");

        } catch (IOException e) {
            System.err.println(e.getMessage());
        }

    }

    public static void writeModel(RegressionValidations<GradientTreeBoost> regressors, String modelPath) throws IOException {
        Write.object(regressors, Paths.get(modelPath));
    }
    public RegressionValidations<GradientTreeBoost> readModel(String modelPath) throws Exception {

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
                results[i] += prediction[i];
            }
        }

        for (int i = 0; i < size; i++) {
            results[i] /= cvSize;
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

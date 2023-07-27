package models;

import data.*;
import ml.dmlc.xgboost4j.java.*;
import org.apache.avro.generic.*;
import org.apache.parquet.hadoop.*;
import smile.io.*;

import java.io.*;
import java.util.*;


public class XGBoostModel {

    static String trainDataPath = "data/524_v4_1_train.libsvm";
    static String liveDataPath = "data/524_v4_1_live.libsvm";
    static String modelPath = "trained_XGBoost4J.model";

    static boolean processType = true;

    static ReadData readData = new ReadData();

    public static void main(String[] args) throws XGBoostError, IOException {

        if (processType) {

            System.out.println("Generating XGBoost parameters");
            Map<String, Object> param = generateXGBParam(20, "reg:linear", 100, 2);

            System.out.println("Reading libsvm file");
            List<DMatrix> dataset = readLibsvm();
            DMatrix trainMatrix = dataset.get(0);
            DMatrix testMatrix = dataset.get(1);

            System.out.println("Training model");
            Booster booster = trainXGBoost(trainMatrix, testMatrix, param, 3);

            System.out.println("Writing a trained model");
            writeModel(booster, modelPath);
        } else {
            System.out.println("Reading libsvm file");
            DMatrix liveMatrix = new DMatrix(liveDataPath);

            Booster booster = XGBoost.loadModel(modelPath);
            float[][] results = booster.predict(liveMatrix);

            List<GenericRecord> recordList = new ArrayList<>();
            ParquetReader<GenericRecord> iter = readData.readAsIterator(liveDataPath);


        }
    }

    public static List<Integer> generateIndex(int size) {
        ArrayList<Integer> indexList = new ArrayList<>(size);

        for (int i = 0; i < size; i++) {
            indexList.add(i);
        }

        Collections.shuffle(indexList);
        return indexList;
    }

    public static List<DMatrix> readLibsvm() throws XGBoostError {
        DMatrix dataMatrix = new DMatrix(trainDataPath);

        int size = (int) dataMatrix.rowNum();
        int trainSize = (int) (size * 0.8);

        List<Integer> indexList = generateIndex(size);

        DMatrix trainMatrix = dataMatrix.slice(indexList.subList(0, trainSize).stream().mapToInt(i->i).toArray());

        DMatrix testMatrix = dataMatrix.slice(indexList.subList(trainSize, size).stream().mapToInt(i->i).toArray());

        ArrayList<DMatrix> dataset = new ArrayList<>();
        dataset.add(trainMatrix);
        dataset.add(testMatrix);

        return dataset;

    }

    public static Map<String, Object> generateXGBParam(int maxDepth, String objective, int numRound, int numWorkers) {
        Map<String, Object> xgbParam = new HashMap<String, Object>() {
            {
                put("max_depth", maxDepth);
                put("objective", objective);
                put("num_round", numRound);
                put("num_workers", numWorkers);
            }
        };
        return xgbParam;
    }

    public static Booster trainXGBoost(DMatrix trainMat, DMatrix testMat, Map<String, Object> params, int nround) throws XGBoostError {
        Map<String, DMatrix> watches = new HashMap<String, DMatrix>() {
            {
                put("train", trainMat);
                put("test", testMat);
            }
        };

        return XGBoost.train(trainMat, params, nround, watches, null, null);
    }

    public static void writeModel(Booster booster, String filename) throws XGBoostError {
        booster.saveModel(filename);
    }


}

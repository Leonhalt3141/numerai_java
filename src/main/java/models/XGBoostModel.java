package models;

import data.ReadDataWithSpark;
import ml.dmlc.xgboost4j.java.XGBoostError;
import ml.dmlc.xgboost4j.scala.spark.XGBoostRegressor;
import ml.dmlc.xgboost4j.scala.spark.XGBoostRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.*;

import java.io.IOException;
import java.util.HashMap;
//import java.util.Map;

import scala.collection.immutable.Map;


public class XGBoostModel {

    static String trainParquetPath = "data/509_v4_1_train.parquet";

    public static void main(String[] args) throws IOException, XGBoostError {
        StructType schema = ReadDataWithSpark.getSchema(trainParquetPath);

        String[] inputNames = ReadDataWithSpark.getInputNames(schema);

        Dataset<Row> inputVector = ReadDataWithSpark
                .convertToInputVector(schema, "output", inputNames, "target", trainParquetPath);

        Map<String, Object> param = generateXGBParam(20, "multi:softprob", 100, 2);

        XGBoostRegressionModel model = train(inputVector, param);

        writeModel(model, "trained_XGBoost4J.model");
    }

    public static Map<String, Object> generateXGBParam(int maxDepth, String objective, int numRound, int numWorkers) {
        Map<String, Object> xgbParam = new scala.collection.immutable.HashMap<>();
        xgbParam.updated("max_depth", maxDepth);
        xgbParam.updated("objective", objective);
        xgbParam.updated("num_round", numRound);
        xgbParam.updated("num_workers", numWorkers);
        return xgbParam;
    }

    public static XGBoostRegressionModel train(Dataset<Row> xgbInput, Map<String, Object> params) throws IOException, XGBoostError {

        XGBoostRegressor xgbRegressor = new XGBoostRegressor(params)
                .setFeaturesCol("features")
                .setLabelCol("target");

        return xgbRegressor.fit(xgbInput);
    }

    public static void writeModel(XGBoostRegressionModel model, String filename) throws IOException {
        model.write().overwrite().save(filename);
    }


}

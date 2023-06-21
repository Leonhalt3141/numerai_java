package models;

import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoostError;
import ml.dmlc.xgboost4j.scala.spark.XGBoostRegressor;
import ml.dmlc.xgboost4j.scala.spark.XGBoostRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import java.io.IOException;
import java.util.HashMap;
//import java.util.Map;

import scala.collection.immutable.Map;

public class XGBoostModel {

    Map<String, Object> generateXGBParam(Double maxDepth, String objective, int numRound, int numWorkers) {
        Map<String, Object> xgbParam = new scala.collection.immutable.HashMap<>();
        xgbParam.updated("max_depth", maxDepth);
        xgbParam.updated("objective", objective);
        xgbParam.updated("num_round", numRound);
        xgbParam.updated("num_workers", numWorkers);
        return xgbParam;
    }

    XGBoostRegressionModel train(Dataset<Row> xgbInput, Map<String, Object> params) throws IOException, XGBoostError {

        XGBoostRegressor xgbRegressor = new XGBoostRegressor(params)
                .setFeaturesCol("features")
                .setLabelCol("target");

        return xgbRegressor.fit(xgbInput);
    }
}

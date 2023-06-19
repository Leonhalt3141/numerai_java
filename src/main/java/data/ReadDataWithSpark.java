package data;

import java.io.*;

import org.apache.spark.sql.SparkSession;
import smile.data.type.*;

import java.util.*;


public class ReadDataWithSpark {


    StructType createSchema(String filePath) {

        List<StructField> fieldList = new ArrayList<>();

        try{
           BufferedReader br = new BufferedReader(new FileReader(filePath));

           String[] columns = br.readLine().split(",");
           DataType dataType;

            for (String column : columns) {
                if (column.contains("feature")) {
                    dataType = DataTypes.DoubleType;
                } else if (column.contains("target")) {
                    dataType = DataTypes.DoubleType;
                } else {
                    dataType = DataTypes.StringType;
                }

                fieldList.add(new StructField(column, dataType));
            }
            return fieldList;

        } catch (Exception e) {
            e.printStackTrace();
        }

    }

}

package data;

import java.io.*;

import org.apache.parquet.hadoop.util.HadoopInputFile;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;

import smile.data.type.*;
import org.apache.parquet.hadoop.ParquetReader;
import org.apache.avro.generic.GenericRecord;
import org.apache.parquet.avro.*;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;

import java.util.*;


import static org.apache.parquet.avro.AvroParquetReader.builder;


public class ReadDataWithSpark {


    List<StructField> createSchema(String filePath) {

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


        } catch (Exception e) {
            e.printStackTrace();
        }
        return fieldList;
    }

    public ParquetReader<GenericRecord> buildParquetReader(String filePath) throws IOException {
        Configuration conf = new Configuration();

        conf.setBoolean(AvroReadSupport.AVRO_COMPATIBILITY, true);

        Path input = new Path(filePath);

        return AvroParquetReader
                .<GenericRecord>builder(HadoopInputFile.fromPath(input, conf))
                .withConf(conf)
                .build();
    }

    private Map<String, Object> recordToMap(GenericRecord record, List<StructField> fieldList) {
        Map<String, Object> recordMap = new HashMap<>();

        for (StructField field : fieldList) {
            Object value = (!record.hasField(field.name) || record.get(field.name) == null) ? Double.NaN : record.get(field.name);

            recordMap.put(field.name, value);
        }

        return recordMap;
    }

    public Dataset<Row> convertToDMatrix(String schema, String[] inputNames, String csvPath) {
        SparkSession spark = SparkSession.builder().getOrCreate();
        Dataset<Row> rawInput = spark.read().schema(schema).csv(csvPath);

        StringIndexerModel stringIndexer = new StringIndexer()
                .setInputCols(inputNames)
                .setOutputCol("")
                .fit(rawInput);

        Dataset<Row> labelTransformed = stringIndexer.transform(rawInput).drop("");

        VectorAssembler vectorAssembler = new VectorAssembler().
                setInputCols(inputNames).
                setOutputCol("features");

        return vectorAssembler.transform(labelTransformed).select("features", "target");
    }

}

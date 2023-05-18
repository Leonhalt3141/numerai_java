package data;

import org.apache.avro.*;
import smile.data.*;
import org.apache.avro.generic.GenericRecord;
import org.apache.parquet.avro.*;
import org.apache.hadoop.conf.Configuration;
import org.apache.parquet.hadoop.ParquetReader;
import org.apache.hadoop.fs.Path;
import smile.data.type.*;

import java.io.*;
import java.util.*;
import java.util.stream.*;

public class ReadData {

    public ParquetReader<GenericRecord> readAsIterator(String pathStr) throws IOException {
        Configuration conf = new Configuration();
        conf.setBoolean(AvroReadSupport.AVRO_COMPATIBILITY, true);

        Path input = new Path(pathStr);

        //noinspection deprecation
        return AvroParquetReader.<GenericRecord>builder(input)
                        .withConf(conf)
                        .build();

    }

    private Map<String, Object> rowToMap(GenericRecord row, List<String> fieldNames) {
        Map<String, Object> recordMap = new HashMap<>();
        for (String name : fieldNames) {
            Object  value = (row.get(name) == null || !row.hasField(name)) ? null : row.get(name);
            if (value != null) {
                recordMap.put(name, value);
            }
        }
        return recordMap;
    }

    private void addStructField(Schema.Field field, List<String> fieldNames, List<StructField> structArray) {
        String name = field.name();

        fieldNames.add(name);

        DataType dataType;

        if (name.contains("target")) {
            dataType = DataTypes.DoubleType;
        } else if (name.contains("feature")) {
            dataType = DataTypes.DoubleType;
        } else if (name.contains("era")) {
            dataType = DataTypes.StringType;
        } else {
            dataType = DataTypes.StringType;
        }

        structArray.add(new StructField(name, dataType));
    }

    public DataFrame convertToDataFrame(List<GenericRecord> recordList) {
        ListIterator<Schema.Field> fieldIter = recordList.get(0).getSchema().getFields().listIterator();
        List<String> fieldNames = new ArrayList<>();

        List<StructField> structArray = new ArrayList<>();

        while (fieldIter.hasNext()) {
            addStructField(fieldIter.next(), fieldNames, structArray);
        }

        StructType schema = DataTypes.struct(structArray);

        Collection<Map<String, Object>> data = recordList.stream().map(row -> rowToMap(row, fieldNames))
                .collect(Collectors.toList());
        return DataFrame.of(data, schema);
    }

}

package task1;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

public class Task1Reducer extends Reducer<Text, Text, Text, Text> {
    Text result = new Text();


    public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {

        Configuration conf = context. getConfiguration();
        String country_1 = conf.get("country1");
        String country_2 = conf.get("country2");
        List<String> videolist = new ArrayList<String>();
        Map<String,List<String>> overlapMap = new HashMap<String,List<String>>();

        double count_country1 = 0;
        double count_overlap = 0;

        for (Text text: values){
            String id_country = text.toString();
            String[] dataArray = id_country.split(",");
            if (dataArray[1].equals(country_1) && !videolist.contains(dataArray[0])){
                count_country1++;
                videolist.add(dataArray[0]);
            }
//           if the country of this match the country_1, the number of video add 1
            if (overlapMap.containsKey(dataArray[0])){
                List<String> country_test = overlapMap.get(dataArray[0]);
                if(!country_test.contains(dataArray[1])){
                    country_test.add(dataArray[1]);
                    overlapMap.put(dataArray[0],country_test);
                }

            }else{
                List<String> list_new = new  ArrayList<String>();
                list_new.add(dataArray[1]);
                overlapMap.put(dataArray[0],list_new);
            }


//           if the list of the video already have this video, then the overlap value add 1
        }

        for (Map.Entry<String,List<String>> entry:overlapMap.entrySet()){
            List<String> countlist = entry.getValue();
            int size = countlist.size();
            if (size == 2){
                count_overlap++;
            }
        }

        double lap_percent = count_overlap/count_country1;
        DecimalFormat df = new DecimalFormat("#.0");

        if(count_country1!= 0){
            StringBuffer strBuf = new StringBuffer();
            strBuf.append("total "+count_country1+"; "+df.format(lap_percent*100)+"% in "+country_2);

            result.set(strBuf.toString());


            context.write(key, result);
        }


    }

}

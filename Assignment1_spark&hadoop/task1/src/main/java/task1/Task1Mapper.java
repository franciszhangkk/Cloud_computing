package task1;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.charset.CharsetEncoder;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

public class Task1Mapper extends Mapper<Object, Text, Text, Text> {
    private Text category = new Text(), video_detail = new Text();

    // a mechanism to filter out non ascii tags
    static CharsetEncoder asciiEncoder = Charset.forName("US-ASCII").newEncoder();

    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {


        Configuration conf = context. getConfiguration();
        String country_1 = conf.get("country1");
        String country_2 = conf.get("country2");
        List<String> countrylist = new ArrayList<String>();
        countrylist.add(country_1);
        countrylist.add(country_2);

        String[] dataArray = value.toString().split(",(?=([^\"]*\"[^\"]*\")*[^\"]*$)"); //split the data into array

        String categoryString = dataArray[5];
        String idString = dataArray[0];
        String countryString = dataArray[17];

        if(countrylist.contains(countryString)){
            String id_country = idString+","+countryString;

            category.set(categoryString);
            video_detail.set(id_country);

            context.write(category,video_detail);
        }else {
            return;
        }

    }
}

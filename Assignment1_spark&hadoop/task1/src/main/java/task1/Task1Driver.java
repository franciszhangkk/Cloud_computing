package task1;
import java.io.IOException;
import java.lang.*;
import com.sun.tools.internal.ws.wsdl.document.jaxws.Exception;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

public class Task1Driver {
    public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException {

        Configuration conf = new Configuration();
        String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
        if (otherArgs.length != 4) {
            System.err.println("Usage: TagDriver <in> <out> <Country_1> <Country_2>");
            System.exit(2);
        }
        String country_1 = otherArgs[2];
        String country_2 = otherArgs[3];

        conf.set("country1",country_1);
        conf.set("country2",country_2);

        Job job = new Job(conf, "count category overlap");
        job.setNumReduceTasks(3); // we use three reducers, you may modify the number
        job.setJarByClass(Task1Driver.class);
        job.setMapperClass(Task1Mapper.class);
        job.setReducerClass(Task1Reducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        TextInputFormat.addInputPath(job, new Path(otherArgs[0]));
        TextOutputFormat.setOutputPath(job, new Path(otherArgs[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }

}

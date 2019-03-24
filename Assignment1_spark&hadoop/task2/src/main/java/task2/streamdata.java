package task2;

import java.text.DecimalFormat;
import java.util.*;

import com.google.common.collect.Lists;
import org.apache.spark.api.java.*;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.function.Function;

import scala.Tuple2;

public class streamdata {
    public static void main(String[] args) {

        //The program arguments are input and output path
        //Using absolute path is always preferred
        //For windows system, the path value should be something like "C:\\data\\ml-100k\\"
        //For unix system, the path value should something like "/home/user1/data/ml-100k/"
        //For HDFS, the path value should be something like "hdfs://localhost/user/abcd1234/movies/"

        String inputDataPath = args[0], outputDataPath = args[1];
        SparkConf conf = new SparkConf();

        conf.setAppName("Impact of Trending on View Number");

        JavaSparkContext sc = new JavaSparkContext(conf);

        JavaRDD<String> trendData = sc.textFile(inputDataPath);
        //read ratings.csv and convert it to a key value pair RDD of the following format

        JavaPairRDD<String, String> videoExtraction = trendData.mapToPair(s ->
                {  String[] values = s.split(",(?=([^\"]*\"[^\"]*\")*[^\"]*$)");

                    String Country = values[17];
                    String VideoID = values[0];
                    String Trending = values[1];
                    String views = values[8];
                    String key = Country+","+VideoID;
                    String value = Trending+","+views;

                    return
                            new Tuple2<String, String>(key,value);
                }
        );
        //Group the video information by the VideoID

        JavaPairRDD<String,Iterable<String>> GroupedVideo = videoExtraction.groupByKey();

        JavaPairRDD<SecondSortKey,String> Result_video = GroupedVideo.mapToPair(l ->
                {
                    List<String> videos = Lists.newArrayList(l._2);
                    String Country_id = l._1;
                    List<String> list_first2 = new  ArrayList<String>();
                    String[] key_infor = Country_id.split(",");
                    String country = key_infor[0];
                    String video_id = key_infor[1];
                    double num_of_percent = 0;

                    for (String video:videos){//遍历数组里每一个video开始处理
                        String[] videoInfor = video.split(",");
                        String trending_1,views,trending_views,trending_2;
                        trending_1 = videoInfor[0];
                        views = videoInfor[1];
                        String [] trending_dates = trending_1.split("\\.");
                        if (trending_dates.length == 3){
                            trending_2 = trending_dates[0]+trending_dates[2]+trending_dates[1];
                        }else {
                            continue;
                        }
                        int trend_int = Integer.parseInt(trending_2);
                        //处理一下日期的格式方便之后比较
                        trending_views = trending_2+","+views;
                        if (list_first2.size() == 0){
                            list_first2.add(trending_views);
                        }else if(list_first2.size() == 1){
                            String first_trend = list_first2.get(0);
                            String [] first_list = first_trend.split(",");
                            int first_date = Integer.parseInt(first_list[0]);
                            if (trend_int<first_date){
                                list_first2.add(0,trending_views);
                            }else {
                                list_first2.add(trending_views);
                            }
                        }else if (list_first2.size() ==2){
                            String first_trend = list_first2.get(0);
                            String [] first_list = first_trend.split(",");
                            int first_date = Integer.parseInt(first_list[0]);
                            String second_trend = list_first2.get(1);
                            String [] second_list = second_trend.split(",");
                            int second_date = Integer.parseInt(second_list[0]);
                            if(trend_int>second_date){
                                continue;
                            }else if(trend_int>first_date){
                                list_first2.set(1,trending_views);
                            }else {
                                list_first2.set(1,first_trend);
                                list_first2.set(0,trending_views);
                            }
                        }
                    }

                    if (list_first2.size()==2){
                        String string_early = list_first2.get(0);
                        String string_late = list_first2.get(1);
                        String [] early_list = string_early.split(",");
                        String [] late_list = string_late.split(",");
                        double early_views = Long.valueOf(early_list[1]);
                        double late_views = Long.valueOf(late_list[1]);

                        num_of_percent = (late_views-early_views)/early_views;

                    }else if (list_first2.size() == 1){
                        num_of_percent = 0;
                    }

                    double nString = num_of_percent*100;
                    long longvalue = (long) nString;
                    SecondSortKey ssk = new SecondSortKey(country, longvalue);
                    DecimalFormat df = new DecimalFormat("#.0");
                    String output = country+";  "+video_id+",  "+df.format(nString)+"%";
                    return new Tuple2<SecondSortKey, String>(ssk, output);
                }
        );

        JavaPairRDD<SecondSortKey, String> sortByKeyRDD =Result_video.sortByKey(false);
        JavaPairRDD<SecondSortKey, String> filtered = sortByKeyRDD.filter(l ->{
            SecondSortKey ssk = l._1;
            long percent = ssk.getSecond();
            return percent >= 1000;

        });

        JavaRDD<String> result = filtered.map(l ->{
            String value = l._2;
            return value;
        });

        result.saveAsTextFile(outputDataPath + "task2");
        sc.close();
    }
}

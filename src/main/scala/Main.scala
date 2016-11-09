import org.apache.spark.ml.feature._
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.SparkSession

import scala.xml.XML


object Main {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder()
      .master("local[*]")
      .appName("SentimentAnalysis")
      .getOrCreate()



    //Read XML
    val xml = XML.loadFile("src/main/resources/general-tweets-train-tagged.xml")
    //Access to element tweets
    val tweets = (xml \\ "tweets")

    //Create a Dataframe from XML tweets
    val sentenceDataFrame = spark.createDataFrame(
      //Map all tweet elements
      (tweets \\ "tweet").map(
        //return (language, content) fields from tweet element
        tweet => (((tweet \\ "sentiments" \\ "polarity").head \\ "value").text match {
                                                                          case "NONE" => 0.0
                                                                          case "N" => 1.0
                                                                          case "N+" => 2.0
                                                                          case "NEU" => 3.0
                                                                          case "P" => 4.0
                                                                          case "P+" => 5.0
                                                                          case _ => 0.0
                                                                        }
                , (tweet \\ "content").text)
      )
    ).toDF("polarity", "sentence") //Columns of DataFrame

    //Regex Tokenizer (Custom tokenizer)
    val word = "[a-zA-ZñÑáéíóú]" //Simbols that contain a word
    val regexTokenizer = new RegexTokenizer()
      .setInputCol("sentence")
      .setOutputCol("words")
      .setToLowercase(true)
      .setPattern(
      s"(([a-z0-9_\\.-]+)@([\\da-z\\.-]+)\\.([a-z\\.]{2,6}))" + //Emails
        s"|" +
        s"((https?:\\/\\/)([\\da-z\\.-]+)\\.([a-z\\.]{2,6})([\\/\\w \\.-]*)*\\/?)" + //URLs
        s"|" +
        s"(#$word+)" + //Hashtags
        s"|" +
        s"(@$word+)" + //Mentions
        s"|" +
        s"($word+)" + //Words
        s"|" +
        s"([0-9]+)").setGaps(false) //Mentions


    //Create trigram
    val ngram = new NGram().setN(3).setInputCol("words").setOutputCol("ngrams")

    val ngramDataFrame = ngram.transform(
      regexTokenizer.transform(sentenceDataFrame).select("words", "polarity") //Parse with RegexTokenizer and select the words column as a DataFrame
    )
    //ngramDataFrame.select("ngrams").show(false)
    //ngramDataFrame.select("ngrams", "polarity").take(50).foreach(println)

    val htf = new HashingTF().setInputCol("ngrams").setOutputCol("rawFeatures")//.setNumFeatures(500)
    val tf = htf.transform(ngramDataFrame.select("ngrams", "polarity"))
    tf.cache()
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(tf)
    val tfidf = idfModel.transform(tf)

    println()
    tfidf.select("features").take(100).foreach(println)

    import spark.implicits._

    val data = tfidf.map(row =>
      org.apache.spark.mllib.regression.LabeledPoint(
        row.getAs[Double]("polarity"),
        org.apache.spark.mllib.linalg.Vectors.dense(
          row.getAs[org.apache.spark.ml.linalg.SparseVector]("features").values
        )
      )
    ).rdd


    val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    val model = SVMWithSGD.train(training, 100)

    // Clear the default threshold.
    model.clearThreshold()

    // Compute raw scores on the test set.
    val scoreAndLabels = test.map { point =>
      val score = model.predict(point.features)
      (score, point.label)
    }

    // Get evaluation metrics.
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics.areaUnderROC()

    println("Area under ROC = " + auROC)

    // Save and load model
    model.save(spark.sparkContext, "target/tmp/scalaSVMWithSGDModel")
    val sameModel = SVMModel.load(spark.sparkContext, "target/tmp/scalaSVMWithSGDModel")


    spark.stop()
  }
}

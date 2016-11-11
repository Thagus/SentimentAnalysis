import org.apache.spark.sql.SparkSession

import scala.xml.XML


object Main {
  val word = "[a-zA-ZñÑáéíóú]"//Symbols that contain a word
  val pattern =  s"$word+".r  //create word pattern regex

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

    val ts = (tweets \\ "tweet").map(
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
        , clearText((tweet \\ "content").text.toLowerCase()))
    ).filter(tweet => tweet._2.length>0)  //Filter those that have no characters on them


    ts.foreach(tweet => {
      println(tweet._1)

      //Create trigrams
      val ngrams = tweet._2.split(" ")
        .sliding(3) //Slide in trigrams
        .foreach(p => { //For each trigram
          val trigramTerm = p.mkString(" ")
          print(trigramTerm + ", ")

        })

      println()
    })

    spark.stop()
  }
  
   def clearText(document: String) : String = {
     var clearDocument = document.replaceAll("(https?:\\/\\/)([\\da-z\\.-]+)\\.([a-z\\.]{2,6})([\\/\\w \\\\.-]*)*\\/?", "")
     clearDocument = clearDocument.replaceAll("([a-z0-9_\\.-]+)@([\\da-z\\.-]+)\\.([a-z\\.]{2,6})", "")
     clearDocument = clearDocument.replaceAll(s"@$word+", "")
     clearDocument = clearDocument.replaceAll("[0-9]+", "")

     clearDocument = (pattern findAllIn clearDocument).mkString(" ")

     clearDocument
  }
  
}

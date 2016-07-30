package com.mediav.antifraud

import org.apache.spark.graphx.{Edge, Graph, PartitionStrategy}
import org.apache.spark.{SparkConf, SparkContext}
import org.slf4j.LoggerFactory
import scopt.OptionParser

/**
  * Created by wenmin on 16/7/25.
  * An object which implements co-visitation clustering with Graphx
  * The algorithm is given in "Using Co-Visitation Networks For Classifying Non-Intentional Traffic"
  */
object CoVisitationClustering {

  case class CmdConfig(input: String = null, output: String = null, modelName: String = null,
                       numPartition: Int = 100, threshold: Double = 0.5)

  def genParser(): OptionParser[CmdConfig] = {
    new OptionParser[CmdConfig]("CoVisitationClustering") {
      opt[String]("input") required() action { (x, c) => c.copy(input = x) } text
          ("Location of input file")
      opt[String]("output") required() action { (x, c) => c.copy(output = x) } text
          ("Location of ouput file")
      opt[String]("model-name") required() action { (x, c) => c.copy(modelName = x) } text
          ("model name")
      opt[Int]("num-partition") required() action { (x, c) => c.copy(numPartition = x) } text
          ("partition number")
      opt[Double]("threshold") required() action { (x, c) => c.copy(threshold = x) } text
          ("threshold to construct an edge")
      help("help") text ("Usage:")
    }
  }

  def main(args: Array[String]) {
    val logger = LoggerFactory.getLogger(getClass);
    val parser = genParser()
    parser.parse(args, CmdConfig()) map {
      config => {
        logger.info("{}", config)
        val conf = new SparkConf().setAppName(config.modelName)
        val sc = new SparkContext("yarn-cluster", config.modelName, conf)
        val interactions = sc.textFile(config.input).coalesce(config.numPartition).map { line =>
          line.split("\\s+") }.map { arr => (arr(0), arr(1).toInt) }.distinct().cache()
        val visitorsCnt = interactions.map { pair => (pair._2, 1) }.reduceByKey(_ + _)
            .collectAsMap()
        val bVisitorsCnt = sc.broadcast(visitorsCnt)
        val edgePairs = interactions.join(interactions).map {p => (p._2, 1)}.reduceByKey(_ + _).map
        {p => (p._1._1, p._1._2, p._2)}.filter {
          t => (t._1 != t._2) && (t._3 >= (bVisitorsCnt.value(t._1).toDouble max
              bVisitorsCnt.value(t._3)) * config.threshold) }.cache()
        logger.info("=========== Save Edges ===========")
        edgePairs.map { pair => pair._1 + "\t" + pair._2 }.saveAsTextFile(config.output + "/edges")
        interactions.unpersist()
        val edges = edgePairs.map { p => Edge(p._1, p._2, 1) }
        val graph = Graph.fromEdges(edges, 1).partitionBy(PartitionStrategy.RandomVertexCut).cache()
        val cc = graph.connectedComponents().vertices
        logger.info("============= Save Connected Components ===========")
        cc.map { case (vId, cId) => vId + "\t" + cId }.saveAsTextFile(config.output + "/vid-cid")
        logger.info("========== Save Vertex Degree ==========")
        // Note: the number of neighbors of a vertex equals to half of its degree
        graph.degrees.map { case (vId, degree) => vId + "\t" + degree / 2 }.saveAsTextFile(
          config.output + "/vid-degree")
      }
    } getOrElse {
      parser.showUsage()
    }
  }
}

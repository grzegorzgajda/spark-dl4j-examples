/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package examples.cnn;

import static examples.cnn.NetworkTrainer.normalize1;
import static examples.cnn.NetworkTrainer.seed;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Map;

import org.apache.commons.lang.ArrayUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.FeatureUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import scala.Tuple2;

public class ImagesClassification implements Serializable {

	private static final long serialVersionUID = 1L;

	private static final Logger log = LoggerFactory.getLogger(ImagesClassification.class);

	private static final int NUM_CORES = 8;

	public static void main(String[] args) {

		SparkConf conf = new SparkConf();
		conf.setAppName("Images CNN Classification");
		conf.setMaster(String.format("local[%d]", NUM_CORES));
		conf.set(SparkDl4jMultiLayer.AVERAGE_EACH_ITERATION, String.valueOf(true));

		try (JavaSparkContext sc = new JavaSparkContext(conf)) {

			JavaRDD<String> raw = sc.textFile("data/images-data-rgb.csv");
			String first = raw.first();

			JavaPairRDD<String, String> labelData = raw
				.filter(f -> f.equals(first) == false)
				.mapToPair(r -> {
					String[] tab = r.split(";");
					return new Tuple2<>(tab[0], tab[1]);
				});

			Map<String, Long> labels = labelData
				.map(t -> t._1).distinct().zipWithIndex()
				.mapToPair(t -> new Tuple2<>(t._1, t._2))
				.collectAsMap();

			log.info("Number of labels {}", labels.size());
			labels.forEach((a, b) -> log.info("{}: {}", a, b));

			NetworkTrainer trainer = new NetworkTrainer.Builder()
				.model(ModelLibrary.net1)
				.networkToSparkNetwork(net -> new SparkDl4jMultiLayer(sc, net))
				.numLabels(labels.size())
				.cores(NUM_CORES).build();

			JavaRDD<Tuple2<INDArray, double[]>> labelsWithData = labelData
				.map(t -> {
					INDArray label = FeatureUtil.toOutcomeVector(labels.get(t._1).intValue(), labels.size());
					double[] arr = Arrays.stream(t._2.split(" "))
						.map(normalize1)
						.mapToDouble(Double::doubleValue).toArray();
					return new Tuple2<>(label, arr);
				});

			JavaRDD<Tuple2<INDArray, double[]>>[] splited = labelsWithData.randomSplit(new double[] { .8, .2 }, seed);

			JavaRDD<DataSet> testDataset = splited[1]
				.map(t -> {
					INDArray features = Nd4j.create(t._2, new int[] { 1, t._2.length });
					return new DataSet(features, t._1);
				}).cache();
			log.info("Number of test images {}", testDataset.count());

			JavaRDD<DataSet> plain = splited[0]
				.map(t -> {
					INDArray features = Nd4j.create(t._2, new int[] { 1, t._2.length });
					return new DataSet(features, t._1);
				});

			/*
			 * JavaRDD<DataSet> flipped = splited[0].randomSplit(new double[] { .5, .5 }, seed)[0].
			 */
			JavaRDD<DataSet> flipped = splited[0]
				.map(t -> {
					double[] arr = t._2;
					int idx = 0;
					double[] farr = new double[arr.length];
					for (int i = 0; i < arr.length; i += trainer.width) {
						double[] temp = Arrays.copyOfRange(arr, i, i + trainer.width);
						ArrayUtils.reverse(temp);
						for (int j = 0; j < trainer.height; ++j) {
							farr[idx++] = temp[j];
						}
					}
					INDArray features = Nd4j.create(farr, new int[] { 1, farr.length });
					return new DataSet(features, t._1);
				});

			JavaRDD<DataSet> trainDataset = plain.union(flipped).cache();
			log.info("Number of train images {}", trainDataset.count());

			trainer.train(trainDataset, testDataset);
		}
	}

}

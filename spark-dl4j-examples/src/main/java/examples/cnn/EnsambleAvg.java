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
import static examples.utils.Dl4jUtils.asString;
import static examples.utils.Dl4jUtils.label;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.FeatureUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import scala.Tuple2;

public class EnsambleAvg implements Serializable {

	private static final long serialVersionUID = 1L;

	private static final Logger log = LoggerFactory.getLogger(EnsambleAvg.class);

	public static void main(String[] args) {

		SparkConf conf = new SparkConf();
		conf.setMaster("local[*]");
		conf.setAppName("Images CNN Classification Ensamble");
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

			int numLabels = labels.size();
			log.info("Number of labels {}", numLabels);
			labels.forEach((a, b) -> log.info("{}: {}", a, b));

			JavaRDD<Tuple2<INDArray, double[]>> labelsWithData = labelData
				.map(t -> {
					INDArray label = FeatureUtil.toOutcomeVector(labels.get(t._1).intValue(), labels.size());
					double[] arr = Arrays.stream(t._2.split(" "))
						.map(normalize1)
						.mapToDouble(Double::doubleValue).toArray();
					return new Tuple2<>(label, arr);
				});

			JavaRDD<Tuple2<INDArray, double[]>>[] splited = labelsWithData.randomSplit(new double[] { .8, .2 }, seed);

			JavaRDD<DataSet> test = splited[1]
				.map(t -> {
					INDArray features = Nd4j.create(t._2, new int[] { 1, t._2.length });
					return new DataSet(features, t._1);
				});
			log.info("Number of test images {}", test.count());

			String dir = EnsambleAvg.class.getClassLoader().getResource("models").getFile();
			MultiLayerNetwork n1 = ModelSerializer.restoreMultiLayerNetwork(new File(dir, "0.7596314907872697"));
			MultiLayerNetwork n2 = ModelSerializer.restoreMultiLayerNetwork(new File(dir, "0.7763819095477387"));
			MultiLayerNetwork n3 = ModelSerializer.restoreMultiLayerNetwork(new File(dir, "0.7646566164154104"));

			test.filter(
					ds -> label(predictAvg(numLabels, ds, n1, n2, n3)) != label(ds.getLabels()))
				.foreach(
						ds -> log.info("predicted {}, label {}",
								asString(predictAvg(numLabels, ds, n1, n2, n3)),
								label(ds.getLabels()))
				);

			JavaPairRDD<Object, Object> predictionsAndLabels = test
				.mapToPair(
				ds -> new Tuple2<>(label(predictAvg(numLabels, ds, n1, n2, n3)), label(ds.getLabels()))
				);

			MulticlassMetrics metrics = new MulticlassMetrics(predictionsAndLabels.rdd());
			double accuracy = 1.0 * predictionsAndLabels.filter(x -> x._1.equals(x._2)).count() / test.count();
			log.info("accuracy {} ", accuracy);
			predictionsAndLabels.take(10).forEach(t -> log.info("predicted {}, label {}", t._1, t._2));
			log.info("confusionMatrix {}", metrics.confusionMatrix());

		} catch (IOException e) {
			log.error(e.getLocalizedMessage(), e);
		}

	}

	static double[] predictAvg(int numLabels, DataSet ds, MultiLayerNetwork... nets) {

		List<double[]> outputs = Arrays.stream(nets)
			.map(net -> net.output(ds.getFeatureMatrix(), false).data().asDouble())
			.collect(Collectors.toList());

		double[] result = new double[numLabels];
		Arrays.fill(result, 0d);

		outputs.forEach(d -> {
			for (int i = 0; i < numLabels; ++i) {
				result[i] += d[i];
			}
		});

		for (int i = 0; i < numLabels; ++i) {
			result[i] /= nets.length;
		}

		return result;
	}

}

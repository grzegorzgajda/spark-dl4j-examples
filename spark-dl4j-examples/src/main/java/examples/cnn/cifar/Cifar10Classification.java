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
package examples.cnn.cifar;

import static examples.cnn.NetworkTrainer.normalize2;

import java.io.Serializable;
import java.util.Arrays;
import java.util.function.Function;

import org.apache.commons.lang.ArrayUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.input.PortableDataStream;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.FeatureUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import scala.Tuple2;
import examples.cnn.ModelLibrary;
import examples.cnn.NetworkTrainer;
import examples.utils.CifarReader;

public class Cifar10Classification implements Serializable {

	private static final long serialVersionUID = 1L;

	private static final Logger log = LoggerFactory.getLogger(Cifar10Classification.class);

	private static final int NUM_CORES = 8;
	
	private static Function<String, String> extractFileName = s -> s.substring(1 + s.lastIndexOf('/'));

	public static void main(String[] args) {

		CifarReader.downloadAndExtract();
		
		int numLabels = 10;

		SparkConf conf = new SparkConf();
		conf.setMaster(String.format("local[%d]", NUM_CORES));
		conf.setAppName("Cifar-10 CNN Classification");
		conf.set(SparkDl4jMultiLayer.AVERAGE_EACH_ITERATION, String.valueOf(true));

		try (JavaSparkContext sc = new JavaSparkContext(conf)) {

			NetworkTrainer trainer = new NetworkTrainer.Builder()
				.model(ModelLibrary.net2)
				.networkToSparkNetwork(net -> new SparkDl4jMultiLayer(sc, net))
				.numLabels(numLabels)
				.cores(NUM_CORES).build();

			JavaPairRDD<String, PortableDataStream> files = sc.binaryFiles("data/cifar-10-batches-bin");

			JavaRDD<double[]> imagesTrain = files
				.filter(f -> ArrayUtils.contains(CifarReader.TRAIN_DATA_FILES, extractFileName.apply(f._1)))
				.flatMap(f -> CifarReader.rawDouble(f._2.open()));

			JavaRDD<double[]> imagesTest = files
				.filter(f -> CifarReader.TEST_DATA_FILE.equals(extractFileName.apply(f._1)))
				.flatMap(f -> CifarReader.rawDouble(f._2.open()));

			JavaRDD<DataSet> testDataset = imagesTest
				.map(i -> {
					INDArray label = FeatureUtil.toOutcomeVector(Double.valueOf(i[0]).intValue(), numLabels);
					double[] arr = Arrays.stream(ArrayUtils.remove(i, 0)).boxed().map(normalize2)
						.mapToDouble(Double::doubleValue).toArray();
					INDArray features = Nd4j.create(arr, new int[] { 1, arr.length });
					return new DataSet(features, label);
				}).cache();
			log.info("Number of test images {}", testDataset.count());			
			
			JavaPairRDD<INDArray, double[]> labelsWithDataTrain = imagesTrain.mapToPair(
				i -> {
					INDArray label = FeatureUtil.toOutcomeVector(Double.valueOf(i[0]).intValue(), numLabels);
					double[] arr = Arrays.stream(ArrayUtils.remove(i, 0)).boxed().map(normalize2).mapToDouble(Double::doubleValue).toArray();
					return new Tuple2<>(label, arr);
				});

			JavaRDD<DataSet> flipped = labelsWithDataTrain
				.map(t -> {
					double[] arr = t._2;
					int idx = 0;
					double[] farr = new double[arr.length];
					for (int i = 0; i < arr.length; i += trainer.getWidth()) {
						double[] temp = Arrays.copyOfRange(arr, i, i + trainer.getWidth());
						ArrayUtils.reverse(temp);
						for (int j = 0; j < trainer.getHeight(); ++j) {
							farr[idx++] = temp[j];
						}
					}
					INDArray features = Nd4j.create(farr, new int[] { 1, farr.length });
					return new DataSet(features, t._1);
				});

			JavaRDD<DataSet> trainDataset = labelsWithDataTrain
				.map(t -> {
					INDArray features = Nd4j.create(t._2, new int[] { 1, t._2.length });
					return new DataSet(features, t._1);
				}).union(flipped).cache();
			log.info("Number of train images {}", trainDataset.count());	

			trainer.train(trainDataset, testDataset);
		}
	}

}

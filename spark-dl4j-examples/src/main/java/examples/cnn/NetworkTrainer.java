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

import static examples.utils.Dl4jUtils.label;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import scala.Tuple2;
import examples.cnn.ModelLibrary.NetworkModel;
import examples.utils.ModelLoader;

public class NetworkTrainer implements Serializable {
	
	private static final long serialVersionUID = 1L;
	
	public final static int seed = 10205;
	
	public static final Function<String, Double> normalize1 = s -> (Double.parseDouble(s) / 128.0) - 1.0;	
	public static final Function<Double, Double> normalize2 = d -> (d / 128.0) - 1.0;
	
	final Logger log;
	final NetworkModel model;
	final String workingDir;
	final int channels;
	final int height;
	final int width;
	final double minimumLearningRate;
	final double initialLearningRate;
	final double learningRateDecayFactor;
	final double downgradeAccuracyThreshold;
	final int resetLearningRateThreshold;
	final int stepDecayTreshold;
	final int cores;
	final int epochs;
	final int numLabels;
	transient final Function<MultiLayerNetwork, SparkDl4jMultiLayer> networkToSparkNetwork;
	
	private NetworkTrainer(Builder builder){
		log = LoggerFactory.getLogger(getClass());
		model = builder.model;
		workingDir = builder.workingDir;
		channels = builder.channels;
		height = builder.height;
		width = builder.width;
		minimumLearningRate = builder.minimumLearningRate;
		initialLearningRate = builder.initialLearningRate;
		learningRateDecayFactor = builder.learningRateDecayFactor;
		downgradeAccuracyThreshold = builder.downgradeAccuracyThreshold;
		resetLearningRateThreshold = builder.resetLearningRateThreshold;
		stepDecayTreshold = builder.stepDecayTreshold;	
		cores = builder.cores;
		epochs = builder.epochs;
		numLabels = builder.numLabels;
		networkToSparkNetwork = builder.networkToSparkNetwork;
	}
	
	public void train(JavaRDD<DataSet> train, JavaRDD<DataSet> test) {

		int batchSize = 12 * cores;
		int lrCount = 0;
		double bestAccuracy = Double.MIN_VALUE;

		double learningRate = initialLearningRate;

		int trainCount = Long.valueOf(train.count()).intValue();
		log.info("Number of training images {}", trainCount);
		log.info("Number of test images {}", test.count());

		MultiLayerNetwork net = new MultiLayerNetwork(model.apply(learningRate, width, height, channels, numLabels));
		net.init();

		Map<Integer, Double> acc = new HashMap<>();
		for (int i = 0; i < epochs; i++) {

			SparkDl4jMultiLayer sparkNetwork = networkToSparkNetwork.apply(net);
			final MultiLayerNetwork nn = sparkNetwork.fitDataSet(train, batchSize, trainCount, cores);
			log.info("Epoch {} completed", i);

			JavaPairRDD<Object, Object> predictionsAndLabels = test.mapToPair(
					ds -> new Tuple2<>(label(nn.output(ds.getFeatureMatrix(), false)), label(ds.getLabels()))
					);
			MulticlassMetrics metrics = new MulticlassMetrics(predictionsAndLabels.rdd());
			double accuracy = 1.0 * predictionsAndLabels.filter(x -> x._1.equals(x._2)).count() / test.count();
			log.info("Epoch {} accuracy {} ", i, accuracy);
			acc.put(i, accuracy);
			predictionsAndLabels.take(10).forEach(t -> log.info("predicted {}, label {}", t._1, t._2));
			log.info("confusionMatrix {}", metrics.confusionMatrix());

			INDArray params = nn.params();
			if (accuracy > bestAccuracy) {
				bestAccuracy = accuracy;
				try {
					ModelSerializer.writeModel(nn, new File(workingDir, Double.toString(accuracy)), false);
				} catch (IOException e) {
					log.error("Error writing trained model", e);
				}
				lrCount = 0;
			} else {

				if (++lrCount % stepDecayTreshold == 0) {
					learningRate *= learningRateDecayFactor;
				}
				if (lrCount >= resetLearningRateThreshold) {
					lrCount = 0;
					learningRate = initialLearningRate;
				}
				if (learningRate < minimumLearningRate) {
					lrCount = 0;
					learningRate = initialLearningRate;
				}
				if (bestAccuracy - accuracy > downgradeAccuracyThreshold) {
					params = ModelLoader.load(workingDir, bestAccuracy);
				}
			}
			net = new MultiLayerNetwork(model.apply(learningRate, width, height, channels, numLabels));
			net.init();
			net.setParameters(params);
			log.info("Learning rate {} for epoch {}", learningRate, i + 1);
		}
		log.info("Training completed");

	}
	
	public int getHeight() {
		return height;
	}

	public int getWidth() {
		return width;
	}

	public static class Builder{
		NetworkModel model = null;
		String workingDir = "work";
		int channels = 3;
		int height = 32;
		int width = 32;
		double minimumLearningRate = 0.075;
		double initialLearningRate = 0.75;
		double learningRateDecayFactor = 0.8;
		double downgradeAccuracyThreshold = 0.005;
		int resetLearningRateThreshold = 25;
		int stepDecayTreshold = 3;
		Integer cores = null;
		int epochs = 100;		
		Integer numLabels = null;
		Function<MultiLayerNetwork, SparkDl4jMultiLayer> networkToSparkNetwork = null;
		
		public Builder minimumLearningRate(double value){
			minimumLearningRate = value;
			return this;
		}

		public Builder initialLearningRate(double value){
			initialLearningRate = value;
			return this;			
		}

		public Builder learningRateDecayFactor(double value){
			learningRateDecayFactor = value;
			return this;
		}
		
		public Builder downgradeAccuracyThreshold(double value){
			downgradeAccuracyThreshold = value;
			return this;
		}
		public Builder resetLearningRateThreshold(int value){
			resetLearningRateThreshold = value;
			return this;
		}

		public Builder stepDecayTreshold(int value){
			stepDecayTreshold = 3;
			return this;
		}

		public Builder width(int value){
			width = value;
			return this;
		}

		public Builder height(int value){
			height = value;
			return this;
		}

		public Builder channels(int value){
			channels = value;
			return this;
		}

		public Builder workingDir(String value){
			workingDir = value;
			return this;
		}
		
		public Builder model(NetworkModel value){
			model = value;
			return this;
		}		
		
		public Builder cores(int value){
			cores = value;
			return this;
		}

		public Builder epochs(int value){
			epochs = value;
			return this;
		}
		
		public Builder numLabels(int value){
			numLabels = value;
			return this;
		}
		
		public Builder networkToSparkNetwork(Function<MultiLayerNetwork, SparkDl4jMultiLayer> value){
			networkToSparkNetwork = value;
			return this;
		}
		
		public NetworkTrainer build(){
			if(model == null){
				throw new IllegalStateException("Invalid configuration: network model must be specified");
			}
			if(numLabels == null){
				throw new IllegalStateException("Invalid configuration: number of labels must be specified");
			}
			if(cores == null){
				throw new IllegalStateException("Invalid configuration: number of cores available for computations must be specified");
			}
			return new NetworkTrainer(this);
		}
	} 
}

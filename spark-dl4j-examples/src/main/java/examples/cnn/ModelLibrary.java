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

import java.io.Serializable;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LocalResponseNormalization;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.lossfunctions.LossFunctions;


public class ModelLibrary {	

	@FunctionalInterface
	public interface NetworkModel extends Serializable {
		MultiLayerConfiguration apply (double learningRate, int width, int height, int channels, int numLabels);
	}
	
	private static final int seed = 10205;
	
	public static NetworkModel net1 = (learningRate, width, height, channels, numLabels) -> {

		int iterations = 1;

		int layer = 0;
		MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
				.seed(seed)
				.iterations(iterations)
				.regularization(true).l1(0.0001).l2(0.0001)//elastic net regularization
				.learningRate(learningRate)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.updater(Updater.NESTEROVS).momentum(0.9)
				.gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
				.useDropConnect(true)
				.leakyreluAlpha(0.02)
				.list()
				.layer(layer++, new ConvolutionLayer.Builder(3, 3)
						.nIn(channels)
						.padding(1, 1)
						.nOut(64)
						.weightInit(WeightInit.RELU)
						.activation("leakyrelu")
						.build())
				.layer(layer++, new LocalResponseNormalization.Builder().build())
				.layer(layer++, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
						.kernelSize(2, 2)
						.build())

				.layer(layer++, new ConvolutionLayer.Builder(3, 3)
						.padding(1, 1)
						.nOut(64)
						.weightInit(WeightInit.RELU)
						.activation("leakyrelu")
						.build())
				.layer(layer++, new LocalResponseNormalization.Builder().build())
				.layer(layer++, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
						.kernelSize(2, 2)
						.build())

				.layer(layer++, new ConvolutionLayer.Builder(3, 3)
						.padding(0, 0)
						.nOut(64)
						.weightInit(WeightInit.RELU)
						.activation("leakyrelu")
						.build())
				.layer(layer++, new ConvolutionLayer.Builder(3, 3)
						.padding(0, 0)
						.nOut(64)
						.weightInit(WeightInit.RELU)
						.activation("leakyrelu")
						.build())
				.layer(layer++, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
						.kernelSize(2, 2)
						.build())
				.layer(layer++, new DenseLayer.Builder().activation("relu")
						.name("dense")
						.weightInit(WeightInit.NORMALIZED)
						.nOut(384)
						.dropOut(0.5)
						.build())
				.layer(layer++, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
						.nOut(numLabels)
						.weightInit(WeightInit.XAVIER)
						.activation("softmax")
						.build())
				.backprop(true)
				.pretrain(false)
				.cnnInputSize(width, height, channels);
		return builder.build();
	};

	public static NetworkModel net2 = (learningRate, width, height, channels, numLabels) -> {

		int iterations = 1;

		int layer = 0;

		MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
				.seed(seed)
				.iterations(iterations)
				.regularization(true).l1(0.0001).l2(0.0001)
				.learningRate(learningRate)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.updater(Updater.NESTEROVS).momentum(.9)
				.gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
				.useDropConnect(true)
				.leakyreluAlpha(0.02)
				.list()
				.layer(layer++, new ConvolutionLayer.Builder(3, 3)
						.padding(1, 1)
						.nOut(64)
						.weightInit(WeightInit.VI)
						.activation("leakyrelu")
						.build())
				.layer(layer++, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
						.kernelSize(2, 2)
						.build())

				.layer(layer++, new ConvolutionLayer.Builder(3, 3)
						.padding(1, 1)
						.nOut(64)
						.weightInit(WeightInit.VI)
						.activation("leakyrelu")
						.build())
				.layer(layer++, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
						.kernelSize(2, 2)
						.build())

				.layer(layer++, new ConvolutionLayer.Builder(3, 3)
						.padding(0, 0)
						.nOut(64)
						.weightInit(WeightInit.VI)
						.activation("leakyrelu")
						.build())
				.layer(layer++, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
						.kernelSize(2, 2)
						.build())

				.layer(layer++, new LocalResponseNormalization.Builder().build())

				.layer(layer++, new DenseLayer.Builder().activation("relu")
						.name("dense")
						.weightInit(WeightInit.VI)
						.nOut(384)
						.dropOut(0.5)
						.build())
				.layer(layer++, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
						.nOut(numLabels)
						.weightInit(WeightInit.VI)
						.activation("softmax")
						.build())
				.backprop(true)
				.pretrain(false)
				.cnnInputSize(width, height, channels);
		return builder.build();
	};

	public static NetworkModel net3 = (learningRate, width, height, channels, numLabels) -> {

		int iterations = 1;

		int layer = 0;

		MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
				.seed(seed)
				.iterations(iterations)
				.regularization(true).l1(0.0001).l2(0.0001)
				.learningRate(learningRate)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.updater(Updater.NESTEROVS).momentum(0.9)
				.gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
				.useDropConnect(true)
				.leakyreluAlpha(0.02)
				.list()
				.layer(layer++, new ConvolutionLayer.Builder(3, 3)
						.padding(1, 1)
						.nOut(32)
						.weightInit(WeightInit.VI)
						.activation("leakyrelu")
						.build())
				.layer(layer++, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
						.kernelSize(2, 2)
						.build())

				.layer(layer++, new ConvolutionLayer.Builder(3, 3)
						.padding(1, 1)
						.nOut(32)
						.weightInit(WeightInit.VI)
						.activation("leakyrelu")
						.build())
				.layer(layer++, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
						.kernelSize(2, 2)
						.build())

				.layer(layer++, new ConvolutionLayer.Builder(3, 3)
						.padding(0, 0)
						.nOut(64)
						.weightInit(WeightInit.VI)
						.activation("leakyrelu")
						.build())
				.layer(layer++, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
						.kernelSize(2, 2)
						.build())

				.layer(layer++, new LocalResponseNormalization.Builder().build())

				.layer(layer++, new DenseLayer.Builder().activation("relu")
						.name("dense")
						.weightInit(WeightInit.VI)
						.nOut(384)
						.dropOut(.5)
						.build())
				.layer(layer++, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
						.nOut(numLabels)
						.weightInit(WeightInit.VI)
						.activation("softmax")
						.build())
				.backprop(true)
				.pretrain(false)
				.cnnInputSize(width, height, channels);
		return builder.build();
	};

	public static NetworkModel net4 = (learningRate, width, height, channels, numLabels) -> {

		int iterations = 1;

		int layer = 0;

		MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
				.seed(seed)
				.iterations(iterations)
				.regularization(true).l2(0.0005)
				.learningRate(learningRate)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.updater(Updater.NESTEROVS).momentum(.9)
				.gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
				.useDropConnect(true)
				.leakyreluAlpha(0.02)
				.minimize(false)
				.list()
				.layer(layer++, new ConvolutionLayer.Builder(5, 5)
						.nIn(channels)
						.padding(2, 2)
						.nOut(25)
						.weightInit(WeightInit.RELU)
						.activation("leakyrelu")
						.build())
				.layer(layer++, new LocalResponseNormalization.Builder().build())
				.layer(layer++, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
						.kernelSize(2, 2)
						.build())

				.layer(layer++, new ConvolutionLayer.Builder(3, 3)
						.padding(1, 1)
						.nOut(50)
						.weightInit(WeightInit.RELU)
						.activation("leakyrelu")
						.build())
				.layer(layer++, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
						.kernelSize(2, 2)
						.build())
				.layer(layer++, new LocalResponseNormalization.Builder().build())

				.layer(layer++, new DenseLayer.Builder().activation("relu")
						.name("dense")
						.weightInit(WeightInit.NORMALIZED)
						.nOut(400)
						.dropOut(0.5)
						.build())
				.layer(layer++, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
						.nOut(numLabels)
						.weightInit(WeightInit.XAVIER)
						.activation("softmax")
						.build())
				.backprop(true)
				.pretrain(false)
				.cnnInputSize(width, height, channels);
		return builder.build();
	};

}

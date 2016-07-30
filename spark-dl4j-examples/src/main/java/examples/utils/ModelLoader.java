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
package examples.utils;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;

public class ModelLoader {

	public static INDArray load(String workingDir, double ... models) {
		double size = models.length;
		List<INDArray> params = Arrays.stream(models).mapToObj(acc -> {
			try{
				return ModelSerializer.restoreMultiLayerNetwork(new File(workingDir, Double.toString(acc))).params();
			} catch(IOException e){
				throw new RuntimeException(e);
			}
		}).collect(Collectors.toList());
		return params.stream().reduce((m1, m2) -> m1.add(m2)).get().div(size);	
	} 
}

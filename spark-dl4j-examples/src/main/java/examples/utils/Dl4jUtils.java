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

import java.util.ArrayList;
import java.util.List;

import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.nd4j.linalg.api.ndarray.INDArray;

public class Dl4jUtils {
	
	public static double label(INDArray a) {
		int idx = 0;
		double value = Double.MIN_VALUE;
		for (int i = 0; i < a.columns(); ++i) {
			if (a.getDouble(i) > value) {
				idx = i;
				value = a.getDouble(i);
			}
		}
		return idx;
	}

	public static double label(double[] a) {
		int idx = 0;
		double value = Double.MIN_VALUE;
		for (int i = 0; i < a.length; ++i) {
			if (a[i] > value) {
				idx = i;
				value = a[i];
			}
		}
		return idx;
	}

	public static String asString(INDArray a) {
		StringBuilder sb = new StringBuilder();
		sb.append("[");
		for (int i = 0; i < a.columns(); ++i) {
			sb.append(a.getDouble(i)).append(",");
		}
		sb.append("]");
		return sb.toString();
	}

	public static String asString(double[] a) {
		StringBuilder sb = new StringBuilder();
		sb.append("[");
		for (int i = 0; i < a.length; ++i) {
			sb.append(a[i]).append(",");
		}
		sb.append("]");
		return sb.toString();
	}

	public static Vector sv(double[] array) {
		int size = array.length;
		List<Integer> idx = new ArrayList<>();
		for (int i = 0; i < size; ++i) {
			if (array[i] != 0.0) {
				idx.add(i);
			}
		}
		return Vectors.sparse(size, idx.stream().mapToInt(Integer::intValue).toArray(), array);
	}

}

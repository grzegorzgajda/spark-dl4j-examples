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

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.channels.Channels;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.zip.GZIPInputStream;

import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.io.IOUtils;
import org.apache.commons.lang.ArrayUtils;

public class CifarReader {

	public static String ARCHIVE_BINARY_FILE = "cifar-10-binary.tar.gz";
	public static String TEST_DATA_FILE = "test_batch.bin";
	public static String[] TRAIN_DATA_FILES = {
			"data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin", "data_batch_4.bin", "data_batch_5.bin", };

	public static void downloadAndExtract() {

		if (new File("data", TEST_DATA_FILE).exists() == false) {
			try {
				if (new File("data", ARCHIVE_BINARY_FILE).exists() == false) {
					URL website = new URL("http://www.cs.toronto.edu/~kriz/" + ARCHIVE_BINARY_FILE);
					FileOutputStream fos = new FileOutputStream("data/" + ARCHIVE_BINARY_FILE);
					fos.getChannel().transferFrom(Channels.newChannel(website.openStream()), 0, Long.MAX_VALUE);
					fos.close();
				}
				TarArchiveInputStream tar =
						new TarArchiveInputStream(
								new GZIPInputStream(new FileInputStream("data/" + ARCHIVE_BINARY_FILE)));
				TarArchiveEntry entry = null;
				while ((entry = tar.getNextTarEntry()) != null) {
					if (entry.isDirectory()) {
						new File("data", entry.getName()).mkdirs();
					} else {
						byte data[] = new byte[2048];
						int count;
						BufferedOutputStream bos = new BufferedOutputStream(
								new FileOutputStream(new File("data/", entry.getName())), 2048);

						while ((count = tar.read(data, 0, 2048)) != -1) {
							bos.write(data, 0, count);
						}
						bos.close();
					}
				}
				tar.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}

	static int IMAGE_DEPTH = 3;
	static int IMAGE_WIDTH = 32;
	static int IMAGE_HIGHT = 32;
	static int IMAGE_SIZE = 32 * 32;
	
	public static BufferedImage getImageFromArray(double[] pixels, int width, int height) {
		BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
		for (int i = 0; i < pixels.length / IMAGE_DEPTH; ++i) {
			int rgb = new Color(
					Double.valueOf(pixels[i]).intValue(),
					Double.valueOf(pixels[i + 1024]).intValue(),
					Double.valueOf(pixels[i + 2048]).intValue()).getRGB();
			image.setRGB(i % IMAGE_WIDTH, i / IMAGE_HIGHT, rgb);
		}
		return image;
	}
	
	public static List<double[]> rawDouble(String workingDir, String file) {
		try {
			return rawDouble(new BufferedInputStream(new FileInputStream(new File(workingDir, file))));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		return null;
	}

	public static List<double[]> rawDouble(InputStream is) {
		List<double[]> result = new ArrayList<>();
		int row = 1 + (IMAGE_WIDTH * IMAGE_HIGHT * IMAGE_DEPTH);
		try {
			while (is.available() > 0) {
				byte[] arr = new byte[row];
				is.read(arr);
				result.add(Arrays.stream(ArrayUtils.toObject(arr)).mapToDouble(b -> Byte.toUnsignedInt(b)).toArray());
			}
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			IOUtils.closeQuietly(is);
		}
		return result;
	}
	

}

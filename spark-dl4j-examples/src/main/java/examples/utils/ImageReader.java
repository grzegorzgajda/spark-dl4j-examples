package examples.utils;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;

public class ImageReader {

	static int IMAGE_DEPTH = 3;
	static int IMAGE_WIDTH = 32;
	static int IMAGE_HIGHT = 32;
	static int IMAGE_SIZE = 32 * 32;

	public static BufferedImage getImageFromArray(int[] pixels) {

		BufferedImage image = new BufferedImage(IMAGE_WIDTH, IMAGE_HIGHT, BufferedImage.TYPE_INT_RGB);
		for (int i = 0; i < pixels.length / IMAGE_DEPTH; ++i) {
			int rgb = new Color(pixels[i], pixels[i + IMAGE_SIZE], pixels[i + IMAGE_SIZE + IMAGE_SIZE]).getRGB();
			image.setRGB(i % IMAGE_WIDTH, i / IMAGE_HIGHT, rgb);
		}
		return image;
	}

	public static int[] sampleImage(File file) {
		try (BufferedReader br = new BufferedReader(new FileReader(file));) {
			return br.lines().findAny()
				.map(l -> Arrays.stream(l.split(";")[1].split(" ")).mapToInt(Integer::parseInt).toArray()).get();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return null;
	}
}

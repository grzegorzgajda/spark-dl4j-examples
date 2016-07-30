package examples.utils;

import static org.junit.Assert.*;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

import org.junit.Test;

public class ImageReaderTest {

	@Test
	public void testReadSampleImage() {
		int[] pixels = ImageReader.sampleImage(new File("data", "images-data-rgb.csv"));
		assertEquals(pixels.length, ImageReader.IMAGE_DEPTH * ImageReader.IMAGE_SIZE);
	}

	@Test
	public void testCreateBufferedImage() {
		int[] pixels = ImageReader.sampleImage(new File("data", "images-data-rgb.csv"));
		BufferedImage bi = ImageReader.getImageFromArray(pixels);
		assertNotNull(bi);
	}

	@Test
	public void testSaveAsPng() {		
		int[] pixels = ImageReader.sampleImage(new File("data", "images-data-rgb.csv"));
		BufferedImage bi = ImageReader.getImageFromArray(pixels);
		try {
			File png = new File("work", "test.png");
			ImageIO.write(bi, "PNG", png);
			assertTrue(png.exists());
			assertTrue(png.length() > 0);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
}

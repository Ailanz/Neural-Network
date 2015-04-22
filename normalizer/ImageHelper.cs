using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;

namespace NeuralNetwork.normalizer
{
    public class ImageHelper
    {
        public static Bitmap ResizeImage(Image image, int width, int height)
        {
            var destRect = new Rectangle(0, 0, width, height);
            var destImage = new Bitmap(width, height);

            destImage.SetResolution(image.HorizontalResolution, image.VerticalResolution);

            using (var graphics = Graphics.FromImage(destImage))
            {
                graphics.CompositingMode = CompositingMode.SourceCopy;
                graphics.CompositingQuality = CompositingQuality.HighQuality;
                graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
                graphics.SmoothingMode = SmoothingMode.HighQuality;
                graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;

                using (var wrapMode = new ImageAttributes())
                {
                    wrapMode.SetWrapMode(WrapMode.TileFlipXY);
                    graphics.DrawImage(image, destRect, 0, 0, image.Width, image.Height, GraphicsUnit.Pixel, wrapMode);
                }
            }

            return destImage;
        }

        //public static double[] ImageToDoubleArray(Image image)
        //{
        //    MemoryStream ms = new MemoryStream();
        //    image.Save(ms, System.Drawing.Imaging.ImageFormat.Bmp);

        //    byte[] bytes = ms.ToArray();
        //    double[] values = new double[bytes.Length];
        //    for (int i = 0; i < bytes.Length; i++)
        //    {
        //        values[i] = Convert.ToDouble(bytes[i]);
        //    }
        //    return values;
        //}

        //public static Bitmap DoubleArrayToBitmap(double[] data)
        //{
        //    byte[] imageBytes = new byte[data.Length];
        //    for (int i = 0; i < data.Length; i++)
        //    {
        //        imageBytes[i] = Convert.ToByte(data[i]);
        //    }
        //    MemoryStream ms = new MemoryStream(imageBytes);
        //    Image returnImage = Image.FromStream(ms);
        //    return new Bitmap(returnImage);
        //}

        public static double[] ImageToDoubleArray(Bitmap image)
        {
            double[] red = new double[image.Width * image.Height];
            double[] green = new double[image.Width * image.Height];
            double[] blue = new double[image.Width * image.Height];
            int pixelNum = 0;
            for (int i = 0; i < image.Width; i++)
            {
                for (int j = 0; j < image.Height; j++)
                {
                    Color c = image.GetPixel(i, j);
                    red[pixelNum] = Convert.ToDouble(c.R);
                    green[pixelNum] = Convert.ToDouble(c.G);
                    blue[pixelNum] = Convert.ToDouble(c.B);
                    pixelNum++;
                }
            }
            return red.Concat(green).Concat(blue).ToArray(); ;
        }

        public static Bitmap DoubleArrayToBitmap(double[] data, int width, int height)
        {
            if (height * width != data.Length / 3)
            {
                throw new Exception("Dimensions don't match");
            }
            double[] red = new double[data.Length / 3];
            double[] green = new double[data.Length / 3];
            double[] blue = new double[data.Length / 3];
            for (int i = 0; i < red.Length; i++)
            {
                red[i] = data[i];
                green[i] = data[data.Length / 3 + i];
                blue[i] = data[data.Length * 2 / 3 + i];

            }
            int curPos = 0;
            Bitmap image = new Bitmap(width, height);
            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    double curRed = red[curPos];
                    double curGreen = green[curPos];
                    double curBlue = blue[curPos];
                    Color c = Color.FromArgb(Convert.ToByte(curRed), Convert.ToByte(curGreen), Convert.ToByte(curBlue));
                    image.SetPixel(i, j, c);
                    curPos++;
                }
            }
            return image;
        }

        public static Bitmap MakeGrayscale(Bitmap original)
        {
            //create a blank bitmap the same size as original
            Bitmap newBitmap = new Bitmap(original.Width, original.Height);

            //get a graphics object from the new image
            Graphics g = Graphics.FromImage(newBitmap);

            //create the grayscale ColorMatrix
            ColorMatrix colorMatrix = new ColorMatrix(new float[][] 
              {
                 new float[] {.3f, .3f, .3f, 0, 0},
                 new float[] {.59f, .59f, .59f, 0, 0},
                 new float[] {.11f, .11f, .11f, 0, 0},
                 new float[] {0, 0, 0, 1, 0},
                 new float[] {0, 0, 0, 0, 1}
              });

            //create some image attributes
            ImageAttributes attributes = new ImageAttributes();

            //set the color matrix attribute
            attributes.SetColorMatrix(colorMatrix);

            //draw the original image on the new image
            //using the grayscale color matrix
            g.DrawImage(original, new Rectangle(0, 0, original.Width, original.Height),
               0, 0, original.Width, original.Height, GraphicsUnit.Pixel, attributes);

            //dispose the Graphics object
            g.Dispose();
            return newBitmap;
        }


    }
}

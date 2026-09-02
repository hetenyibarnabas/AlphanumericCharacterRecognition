using TorchSharp;
using static TorchSharp.torch;

namespace AlphanumericCharacterRecognition.Data;

public class EmnistDataset
{
    private readonly byte[][] images;
    private readonly byte[] labels;

    public int Count => images.Length;

    public EmnistDataset(string imagePath, string labelPath)
    {
        images = IdxReader.ReadImages(imagePath);
        labels = IdxReader.ReadLabels(labelPath);

        if (images.Length != labels.Length)
        {
            throw new InvalidDataException(
                "The number of images and labels does not match."
            );
        }
    }

    public (Tensor Image, long Label) GetItem(int index)
    {
        if (index < 0 || index >= Count)
        {
            throw new ArgumentOutOfRangeException(nameof(index));
        }

        byte[] source = images[index];

        const int sourceSize = 28;
        const int targetSize = 32;
        const int padding = 2;

        float[] rgb = new float[3 * targetSize * targetSize];

        for (int y = 0; y < sourceSize; y++)
        {
            for (int x = 0; x < sourceSize; x++)
            {
                // EMNIST orientation correction: transpose
                byte pixel = source[x * sourceSize + y];

                // Normalize from 0..255 to 0..1
                float value = pixel / 255.0f;

                int targetX = x + padding;
                int targetY = y + padding;

                int pixelIndex = targetY * targetSize + targetX;

                // R
                rgb[pixelIndex] = value;

                // G
                rgb[targetSize * targetSize + pixelIndex] = value;

                // B
                rgb[2 * targetSize * targetSize + pixelIndex] = value;
            }
        }

        Tensor imageTensor = tensor(rgb)
            .reshape(3, targetSize, targetSize);

        long label = labels[index];

        return (imageTensor, label);
    }
}
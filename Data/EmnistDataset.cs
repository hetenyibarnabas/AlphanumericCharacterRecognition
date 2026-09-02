using TorchSharp;
using static TorchSharp.torch;

namespace AlphanumericCharacterRecognition.Data;

/// <summary>
/// Provides EMNIST images and labels as TorchSharp training samples.
/// </summary>
public class EmnistDataset: ICharacterDataset
{
    private readonly byte[][] images;
    private readonly byte[] labels;

    public int Count => images.Length;

    /// <summary>
    /// Reads IDX image and label files, then validates that they match.
    /// </summary>
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

    /// <summary>
    /// Returns the preprocessed image tensor and label at the given index.
    /// </summary>
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

        // The model expects three 32x32 channels, so pad EMNIST's 28x28 glyphs.
        float[] rgb = new float[3 * targetSize * targetSize];

        for (int y = 0; y < sourceSize; y++)
        {
            for (int x = 0; x < sourceSize; x++)
            {
                // This swaps row/column to correct EMNIST's stored orientation.
                byte pixel = source[x * sourceSize + y];

                // Scale byte pixels to the 0..1 float range used by the network.
                float value = pixel / 255.0f;

                int targetX = x + padding;
                int targetY = y + padding;

                int pixelIndex = targetY * targetSize + targetX;

                // Duplicate grayscale intensity across RGB channels for Conv2d(3, ...).
                rgb[pixelIndex] = value;
                rgb[targetSize * targetSize + pixelIndex] = value;
                rgb[2 * targetSize * targetSize + pixelIndex] = value;
            }
        }

        // TorchSharp uses channel-first image tensors: [channels, height, width].
        Tensor imageTensor = tensor(rgb)
            .reshape(3, targetSize, targetSize);

        long label = labels[index];

        return (imageTensor, label);
    }

    /// <summary>
    /// Returns only the class label for split construction.
    /// </summary>
    public long GetLabel(int index)
    {
        if (index < 0 || index >= Count)
            throw new ArgumentOutOfRangeException(nameof(index));

        return labels[index];
    }
}

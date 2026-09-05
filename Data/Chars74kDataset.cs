using SkiaSharp;
using TorchSharp;

using static TorchSharp.torch;


namespace AlphanumericCharacterRecognition.Data;

/// <summary>
/// Loads Chars74K images as labeled TorchSharp character samples.
/// </summary>
public class Chars74kDataset : ICharacterDataset
{
    private readonly List<(string Path, long Label)> samples = new();

    public int Count => samples.Count;

    /// <summary>
    /// Collects image paths from the Chars74K class folders.
    /// </summary>
    public Chars74kDataset(string rootPath)
    {
        for (int classIndex = 1; classIndex <= 62; classIndex++)
        {
            string directoryName = $"Sample{classIndex:D3}";
            string directoryPath = Path.Combine(rootPath, directoryName);

            if (!Directory.Exists(directoryPath)) continue;

            long label = classIndex - 1;

            foreach (string file in Directory.EnumerateFiles(directoryPath, "*.png"))
            {
                samples.Add((file, label));
            }
        }

        if (samples.Count == 0) throw new InvalidDataException($"No Chars74K images found in: {rootPath}");
    }

    /// <summary>
    /// Loads, resizes, normalizes, and returns one image tensor with its label.
    /// </summary>
   public (Tensor Image, long Label) GetItem(int index)
    {
        if (index < 0 || index >= Count) throw new ArgumentOutOfRangeException(nameof(index));

        var sample = samples[index];

        using var original = SKBitmap.Decode(sample.Path);

        if (original == null) throw new InvalidDataException($"Could not load image: {sample.Path}");

        const int sourceSize = 28;
        const int targetSize = 32;
        const int padding = 2;

        // Resize Chars74K images to EMNIST's 28x28 content size before padding.
        using var resized = new SKBitmap(sourceSize, sourceSize);

        using (var canvas = new SKCanvas(resized))
        {
            canvas.Clear(SKColors.Black);

            var sampling = new SKSamplingOptions( SKFilterMode.Linear, SKMipmapMode.None);

            canvas.DrawBitmap(original,new SKRect(0, 0, sourceSize, sourceSize),sampling);
        }

        float[] rgb = new float[3 * targetSize * targetSize];

        for (int y = 0; y < sourceSize; y++)
        {
            for (int x = 0; x < sourceSize; x++)
            {
                SKColor pixel = resized.GetPixel(x, y);

                int targetX = x + padding;
                int targetY = y + padding;

                int pixelIndex = targetY * targetSize + targetX;

                // Preserve RGB channels and scale byte values to 0..1.
                rgb[pixelIndex] = pixel.Red / 255.0f;

                rgb[targetSize * targetSize + pixelIndex] = pixel.Green / 255.0f;

                rgb[2 * targetSize * targetSize + pixelIndex] = pixel.Blue / 255.0f;
            }
        }

        // Keep the same channel-first shape used by EMNIST samples: [3, 32, 32].
        Tensor tensorImage = tensor(rgb).reshape(3, 32, 32);

        return (tensorImage, sample.Label);
    }


    /// <summary>
    /// Returns only the class label for split construction.
    /// </summary>
    public long GetLabel(int index)
    {
        if (index < 0 || index >= Count) throw new ArgumentOutOfRangeException(nameof(index));

        return samples[index].Label;
    }

}

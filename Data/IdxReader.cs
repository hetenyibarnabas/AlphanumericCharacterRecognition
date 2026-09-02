using System.Buffers.Binary;

namespace AlphanumericCharacterRecognition.Data;

/// <summary>
/// Reads EMNIST image and label files stored in IDX format.
/// </summary>
public static class IdxReader
{
    /// <summary>
    /// Reads an IDX image file as raw flattened pixel data.
    /// </summary>
    public static byte[][] ReadImages(string filePath)
    {
        using var stream = File.OpenRead(filePath);
        using var reader = new BinaryReader(stream);

        // IDX headers store metadata as big-endian 32-bit integers.
        int magicNumber = ReadInt32BigEndian(reader);
        int imageCount = ReadInt32BigEndian(reader);
        int rows = ReadInt32BigEndian(reader);
        int columns = ReadInt32BigEndian(reader);

        if (magicNumber != 2051)
        {
            throw new InvalidDataException(
                $"Invalid image IDX magic number: {magicNumber}"
            );
        }

        int pixelsPerImage = rows * columns;

        // Keep raw flattened images until GetItem applies ML preprocessing.
        var images = new byte[imageCount][];

        for (int i = 0; i < imageCount; i++)
        {
            images[i] = reader.ReadBytes(pixelsPerImage);

            if (images[i].Length != pixelsPerImage)
            {
                throw new EndOfStreamException(
                    $"Unexpected end of file while reading image {i}."
                );
            }
        }

        return images;
    }

    /// <summary>
    /// Reads an IDX label file as an array of class IDs.
    /// </summary>
    public static byte[] ReadLabels(string filePath)
    {
        using var stream = File.OpenRead(filePath);
        using var reader = new BinaryReader(stream);

        // IDX labels use the same big-endian integer header format as images.
        int magicNumber = ReadInt32BigEndian(reader);
        int labelCount = ReadInt32BigEndian(reader);

        if (magicNumber != 2049)
        {
            throw new InvalidDataException(
                $"Invalid label IDX magic number: {magicNumber}"
            );
        }

        // Labels remain class indices; CrossEntropyLoss does not need one-hot targets.
        byte[] labels = reader.ReadBytes(labelCount);

        if (labels.Length != labelCount)
        {
            throw new EndOfStreamException(
                "Unexpected end of file while reading labels."
            );
        }

        return labels;
    }

    /// <summary>
    /// Converts four bytes into an IDX-compatible big-endian integer.
    /// </summary>
    private static int ReadInt32BigEndian(BinaryReader reader)
    {
        byte[] bytes = reader.ReadBytes(4);

        if (bytes.Length != 4)
        {
            throw new EndOfStreamException();
        }

        // BinaryReader follows platform endianness, but IDX files are always big-endian.
        return BinaryPrimitives.ReadInt32BigEndian(bytes);
    }
}

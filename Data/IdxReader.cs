using System.Buffers.Binary;

namespace AlphanumericCharacterRecognition.Data;

public static class IdxReader
{
    public static byte[][] ReadImages(string filePath)
    {
        using var stream = File.OpenRead(filePath);
        using var reader = new BinaryReader(stream);

        int magicNumber = ReadInt32BigEndian(reader);   //Check value at the beggining of the file.
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

        var images = new byte[imageCount][];        //This needs to be optimized, to not load all images into memory at once.

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

    public static byte[] ReadLabels(string filePath)
    {
        using var stream = File.OpenRead(filePath);
        using var reader = new BinaryReader(stream);

        int magicNumber = ReadInt32BigEndian(reader);
        int labelCount = ReadInt32BigEndian(reader);

        if (magicNumber != 2049)
        {
            throw new InvalidDataException(
                $"Invalid label IDX magic number: {magicNumber}"
            );
        }

        byte[] labels = reader.ReadBytes(labelCount);

        if (labels.Length != labelCount)
        {
            throw new EndOfStreamException(
                "Unexpected end of file while reading labels."
            );
        }

        return labels;
    }

    private static int ReadInt32BigEndian(BinaryReader reader)
    {
        byte[] bytes = reader.ReadBytes(4);

        if (bytes.Length != 4)
        {
            throw new EndOfStreamException();
        }

        return BinaryPrimitives.ReadInt32BigEndian(bytes);
    }
}
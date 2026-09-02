namespace AlphanumericCharacterRecognition.Data;

public static class CharacterMapping
{
    public static Dictionary<int, char> Load(string filePath)
    {
        var mapping = new Dictionary<int, char>();

        foreach (string line in File.ReadLines(filePath))
        {
            if (string.IsNullOrWhiteSpace(line))
                continue;

            string[] parts = line.Split(
                ' ',
                StringSplitOptions.RemoveEmptyEntries
            );

            int label = int.Parse(parts[0]);
            int characterCode = int.Parse(parts[1]);

            mapping[label] = (char)characterCode;
        }

        return mapping;
    }
}
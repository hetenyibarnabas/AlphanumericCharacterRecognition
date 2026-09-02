namespace AlphanumericCharacterRecognition.Data;

/// <summary>
/// Loads the mapping between EMNIST labels and characters.
/// </summary>
public static class CharacterMapping
{
    /// <summary>
    /// Converts a mapping file into a label-to-character dictionary.
    /// </summary>
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

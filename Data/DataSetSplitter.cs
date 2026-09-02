namespace AlphanumericCharacterRecognition.Data;

/// <summary>
/// Creates train and test subsets while preserving class balance.
/// </summary>
public static class DatasetSplitter
{
    /// <summary>
    /// Splits a dataset by label so each class keeps the same train/test ratio.
    /// </summary>
    public static (DatasetSubset Train, DatasetSubset Test) StratifiedSplit(
        ICharacterDataset dataset,  double trainRatio = 0.8, int seed = 42)
    {
        if (trainRatio <= 0 || trainRatio >= 1) throw new ArgumentOutOfRangeException(nameof(trainRatio));

        var random = new Random(seed);

        var groups = new Dictionary<long, List<int>>();

        // Group indices by class before splitting to avoid label imbalance.
        for (int i = 0; i < dataset.Count; i++)
        {
            long label = dataset.GetLabel(i);

            if (!groups.ContainsKey(label))  groups[label] = new List<int>();

            groups[label].Add(i);
        }

        var trainIndices = new List<int>();
        var testIndices = new List<int>();

        foreach (var group in groups.Values)
        {
            // Shuffle within each class before assigning train and test samples.
            Shuffle(group, random);

            int trainCount = (int)Math.Floor(group.Count * trainRatio);

            trainIndices.AddRange( group.Take(trainCount));

            testIndices.AddRange( group.Skip(trainCount));
        }

        // Shuffle final subsets so batches are not ordered by class.
        Shuffle(trainIndices, random);
        Shuffle(testIndices, random);

        return (new DatasetSubset(dataset, trainIndices), new DatasetSubset(dataset, testIndices));
    }

    /// <summary>
    /// Randomizes indices with the provided seeded generator.
    /// </summary>
    private static void Shuffle(
        List<int> values,
        Random random)
    {
        for (int i = values.Count - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);

            (values[i], values[j]) = (values[j], values[i]);
        }
    }
}

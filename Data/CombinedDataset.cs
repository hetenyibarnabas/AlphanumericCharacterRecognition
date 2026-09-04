using static TorchSharp.torch;

namespace AlphanumericCharacterRecognition.Data;

/// <summary>
/// Presents multiple character datasets as one continuous dataset.
/// </summary>
public class CombinedDataset : ICharacterDataset
{
    private readonly ICharacterDataset[] datasets;
    private readonly int[] offsets;

    public int Count { get; }

    /// <summary>
    /// Records source datasets and their global index offsets.
    /// </summary>
    public CombinedDataset(params ICharacterDataset[] datasets)
    {
        if (datasets.Length == 0) throw new ArgumentException("At least one dataset is required.");

        this.datasets = datasets;
        offsets = new int[datasets.Length];

        int total = 0;

        for (int i = 0; i < datasets.Length; i++)
        {
            offsets[i] = total;
            total += datasets[i].Count;
        }

        Count = total;
    }

    /// <summary>
    /// Returns a sample from the source dataset that owns the global index.
    /// </summary>
    public (Tensor Image, long Label) GetItem(int index)
    {
        var (dataset, localIndex) = ResolveIndex(index);

        return dataset.GetItem(localIndex);
    }

    /// <summary>
    /// Returns a label from the source dataset that owns the global index.
    /// </summary>
    public long GetLabel(int index)
    {
        var (dataset, localIndex) = ResolveIndex(index);

        return dataset.GetLabel(localIndex);
    }

    /// <summary>
    /// Maps a combined dataset index to a source dataset and local index.
    /// </summary>
    private (ICharacterDataset Dataset, int LocalIndex) ResolveIndex(int index)
    {
        if (index < 0 || index >= Count) throw new ArgumentOutOfRangeException(nameof(index));

        // Offsets keep combined batching independent of each source dataset size.
        for (int i = datasets.Length - 1; i >= 0; i--)
        {
            if (index >= offsets[i])
            {
                return ( datasets[i], index - offsets[i] );
            }
        }

        throw new InvalidOperationException();
    }
}

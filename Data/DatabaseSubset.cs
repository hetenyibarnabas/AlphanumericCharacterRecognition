using static TorchSharp.torch;

namespace AlphanumericCharacterRecognition.Data;

/// <summary>
/// Exposes selected indices from a larger character dataset.
/// </summary>
public class DatasetSubset : ICharacterDataset
{
    private readonly ICharacterDataset source;
    private readonly int[] indices;

    public int Count => indices.Length;

    /// <summary>
    /// Stores the source dataset and the indices included in this subset.
    /// </summary>
    public DatasetSubset( ICharacterDataset source, IEnumerable<int> indices)
    {
        this.source = source;
        this.indices = indices.ToArray();
    }

    /// <summary>
    /// Returns the source sample mapped through this subset's index list.
    /// </summary>
    public (Tensor Image, long Label) GetItem(int index)
    {
        if (index < 0 || index >= Count) throw new ArgumentOutOfRangeException(nameof(index));

        return source.GetItem(indices[index]);
    }

    /// <summary>
    /// Returns the source label mapped through this subset's index list.
    /// </summary>
    public long GetLabel(int index)
    {
        if (index < 0 || index >= Count) throw new ArgumentOutOfRangeException(nameof(index));

        return source.GetLabel(indices[index]);
    }
}

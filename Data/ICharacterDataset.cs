using TorchSharp;
using static TorchSharp.torch;

namespace AlphanumericCharacterRecognition.Data;

/// <summary>
/// Defines the common dataset contract used by training and evaluation.
/// </summary>
public interface ICharacterDataset
{
    int Count { get; }

    /// <summary>
    /// Returns one preprocessed image tensor and its class label.
    /// </summary>
    (Tensor Image, long Label) GetItem(int index);

    /// <summary>
    /// Returns the class label without loading the image tensor.
    /// </summary>
    long GetLabel(int index);
}

using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace AlphanumericCharacterRecognition.Model;

/// <summary>
/// Convolutional neural network for EMNIST character classification.
/// </summary>
public class CharacterRecognitionCNN : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> conv1;
    private readonly Module<Tensor, Tensor> conv2;
    private readonly Module<Tensor, Tensor> pool;
    private readonly Module<Tensor, Tensor> relu;
    private readonly Module<Tensor, Tensor> flatten;
    private readonly Module<Tensor, Tensor> fc1;
    private readonly Module<Tensor, Tensor> fc2;

    /// <summary>
    /// Builds the convolutional, pooling, and fully connected layers.
    /// </summary>
    public CharacterRecognitionCNN() : base("CharacterRecognitionCNN")
    {
        // Padded convolutions preserve spatial size before each pooling step.
        conv1 = Conv2d(3, 32, kernel_size: 3, padding: 1);
        conv2 = Conv2d(32, 64, kernel_size: 3, padding: 1);

        pool = MaxPool2d(kernel_size: 2);
        relu = ReLU();
        flatten = Flatten();

        // Two 2x2 pools reduce 32x32 inputs to 8x8 feature maps.
        fc1 = Linear(64 * 8 * 8, 128);
        // EMNIST ByClass has 62 character classes, the model returns 62 logits.
        fc2 = Linear(128, 62);

        RegisterComponents();
    }

    /// <summary>
    /// Runs the forward pass and returns one logit per class.
    /// </summary>
    public override Tensor forward(Tensor x)
    {
        // Input batches are [batch, channels, height, width] = [N, 3, 32, 32].
        x = conv1.forward(x);
        x = relu.forward(x);
        x = pool.forward(x);

        x = conv2.forward(x);
        x = relu.forward(x);
        x = pool.forward(x);

        // Flatten turns [N, 64, 8, 8] features into [N, 4096] vectors.
        x = flatten.forward(x);
        x = fc1.forward(x);
        x = relu.forward(x);
        // Return raw logits; CrossEntropyLoss handles log-softmax internally.
        x = fc2.forward(x);

        return x;
    }
}

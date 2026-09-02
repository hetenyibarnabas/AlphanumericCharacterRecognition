using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace AlphanumericCharacterRecognition.Model;

public class CharacterRecognitionCNN : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> conv1;
    private readonly Module<Tensor, Tensor> conv2;
    private readonly Module<Tensor, Tensor> pool;
    private readonly Module<Tensor, Tensor> relu;
    private readonly Module<Tensor, Tensor> flatten;
    private readonly Module<Tensor, Tensor> fc1;
    private readonly Module<Tensor, Tensor> fc2;

    public CharacterRecognitionCNN() : base("CharacterRecognitionCNN")
    {
        conv1 = Conv2d(3, 32, kernel_size: 3, padding: 1);
        conv2 = Conv2d(32, 64, kernel_size: 3, padding: 1);

        pool = MaxPool2d(kernel_size: 2);
        relu = ReLU();
        flatten = Flatten();

        fc1 = Linear(64 * 8 * 8, 128);
        fc2 = Linear(128, 62);

        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        x = conv1.forward(x);
        x = relu.forward(x);
        x = pool.forward(x);

        x = conv2.forward(x);
        x = relu.forward(x);
        x = pool.forward(x);

        x = flatten.forward(x);
        x = fc1.forward(x);
        x = relu.forward(x);
        x = fc2.forward(x);

        return x;
    }
}
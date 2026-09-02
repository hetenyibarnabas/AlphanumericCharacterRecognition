using TorchSharp;

var tensor = torch.tensor(new float[]
{
    1, 2, 3,
    4, 5, 6
});

tensor = tensor.reshape(2, 3);

Console.WriteLine("TorchSharp works!");
Console.WriteLine(tensor[0, 0].item<float>());
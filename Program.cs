using TorchSharp;
using AlphanumericCharacterRecognition.Model;
using static TorchSharp.torch;

using var model = new CharacterRecognitionCNN();

using var testImage = randn(1, 3, 32, 32);

using var output = model.forward(testImage);

Console.WriteLine($"Input shape: [{string.Join(", ", testImage.shape)}]");
Console.WriteLine($"Output shape: [{string.Join(", ", output.shape)}]");
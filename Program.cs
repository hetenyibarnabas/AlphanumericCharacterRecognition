using AlphanumericCharacterRecognition.Data;
using AlphanumericCharacterRecognition.Model;
using AlphanumericCharacterRecognition.Training;
using AlphanumericCharacterRecognition.Evaluation;

const string modelPath = "models/characterRecognition_cnn.dat";

if (args.Length == 0)
{
    Console.WriteLine("Usage:");
    Console.WriteLine("  dotnet run -- train");
    Console.WriteLine("  dotnet run -- evaluate");
    return;
}

string command = args[0].ToLower();

switch (command)
{
    case "train":
        Train();
        break;

    case "evaluate":
        Evaluate();
        break;

    default:
        Console.WriteLine($"Unknown command: {command}");
        break;
}


void Train()
{
    var trainDataset = new EmnistDataset(
        "data/emnist/emnist-byclass-train-images-idx3-ubyte",
        "data/emnist/emnist-byclass-train-labels-idx1-ubyte"
    );

    using var model = new CharacterRecognitionCNN();

    var trainer = new Trainer(model, trainDataset);

    trainer.Train(
        epochs: 3,
        batchSize: 64,
        maxSamples: 50000
    );

    Directory.CreateDirectory("models");
    model.save(modelPath);

    Console.WriteLine($"Model saved: {modelPath}");
}



void Evaluate()
{
    if (!File.Exists(modelPath))
    {
        Console.WriteLine("No trained model found.");
        return;
    }

    var testDataset = new EmnistDataset(
        "data/emnist/emnist-byclass-test-images-idx3-ubyte",
        "data/emnist/emnist-byclass-test-labels-idx1-ubyte"
    );

    using var model = new CharacterRecognitionCNN();

    model.load(modelPath);

    var evaluator = new Evaluator(model, testDataset);

    evaluator.Evaluate(
        batchSize: 64,
        maxSamples: null
    );
}
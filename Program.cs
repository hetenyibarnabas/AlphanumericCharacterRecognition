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

    case "train-chars":
        TrainChars74k();
        break;

    case "evaluate-chars":
        EvaluateChars74k();
        break;

    default:
        Console.WriteLine($"Unknown command: {command}");
        break;
}


// Trains and saves the EMNIST model.
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



// Loads the saved EMNIST model and evaluates it on the test set.
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

    evaluator.Evaluate( batchSize: 64, maxSamples: null);
}

// Trains and saves a model using a stratified Chars74K split.
void TrainChars74k()
{
    var fullDataset = new Chars74kDataset(
        "data/chars74k/English/Img/GoodImg/Bmp"
    );

    var (trainSet, _) = DatasetSplitter.StratifiedSplit(
        fullDataset,
        trainRatio: 0.8,
        seed: 42
    );

    using var model = new CharacterRecognitionCNN();

    var trainer = new Trainer(model, trainSet);

    trainer.Train(
        epochs: 3,
        batchSize: 64,
        maxSamples: null
    );

    Directory.CreateDirectory("models");

    model.save("models/chars74k_cnn.dat");

    Console.WriteLine("Chars74K model saved.");
}

// Loads the saved Chars74K model and evaluates it on the held-out split.
void EvaluateChars74k()
{
    const string modelPath = "models/chars74k_cnn.dat";

    if (!File.Exists(modelPath))
    {
        Console.WriteLine("Chars74K model not found.");
        return;
    }

    var fullDataset = new Chars74kDataset(
        "data/chars74k/English/Img/GoodImg/Bmp"
    );

    var (_, testSet) = DatasetSplitter.StratifiedSplit(
        fullDataset,
        trainRatio: 0.8,
        seed: 42
    );

    using var model = new CharacterRecognitionCNN();

    model.load(modelPath);

    var evaluator = new Evaluator(model, testSet);

    evaluator.Evaluate(
        batchSize: 64,
        maxSamples: null
    );
}

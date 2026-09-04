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

    case "test-combined":
        TestCombinedDataset();
        break;

    case "train-combined":
        TrainCombined();
        break;

    case "evaluate-combined":
        EvaluateCombined();
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

    trainer.Train( epochs: 3, batchSize: 64, maxSamples: 50000 );


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

    var (trainSet, _) = DatasetSplitter.StratifiedSplit( fullDataset, trainRatio: 0.8, seed: 42 );

    using var model = new CharacterRecognitionCNN();

    var trainer = new Trainer(model, trainSet);

    trainer.Train( epochs: 3, batchSize: 64, maxSamples: null );

    Directory.CreateDirectory("models");

    model.save("models/chars74kCharRecog_cnn.dat");

    Console.WriteLine("Chars74K model saved.");
}

// Loads the saved Chars74K model and evaluates it on the held-out split.
void EvaluateChars74k()
{
    const string modelPath = "models/chars74kCharRecog_cnn.dat";

    if (!File.Exists(modelPath))
    {
        Console.WriteLine("Chars74K model not found.");
        return;
    }

    var fullDataset = new Chars74kDataset(
        "data/chars74k/English/Img/GoodImg/Bmp"
    );

    var (_, testSet) = DatasetSplitter.StratifiedSplit( fullDataset, trainRatio: 0.8, seed: 42 );

    using var model = new CharacterRecognitionCNN();

    model.load(modelPath);

    var evaluator = new Evaluator(model, testSet);

    evaluator.Evaluate( batchSize: 64, maxSamples: null );
}


// Checks that combined sources produce compatible tensor shapes.
void TestCombinedDataset()
{
    var emnist = new EmnistDataset(
        "data/emnist/emnist-byclass-train-images-idx3-ubyte",
        "data/emnist/emnist-byclass-train-labels-idx1-ubyte"
    );

    var emnistSubset = new DatasetSubset( emnist, Enumerable.Range(0, 50000) );

    var chars74k = new Chars74kDataset(
        "data/chars74k/English/Img/GoodImg/Bmp"
    );

    var (charsTrain, _) = DatasetSplitter.StratifiedSplit(chars74k, trainRatio: 0.8, seed: 42);

    var combined = new CombinedDataset( emnistSubset, charsTrain );

    Console.WriteLine($"EMNIST: {emnistSubset.Count}");
    Console.WriteLine($"Chars74K: {charsTrain.Count}");
    Console.WriteLine($"Combined: {combined.Count}");

    var firstSample = combined.GetItem(0);
    var lastSample = combined.GetItem(combined.Count - 1);

    Console.WriteLine(
        $"First shape: [{string.Join(", ", firstSample.Image.shape)}]"
    );

    Console.WriteLine(
        $"Last shape: [{string.Join(", ", lastSample.Image.shape)}]"
    );

    firstSample.Image.Dispose();
    lastSample.Image.Dispose();
}

// Trains and saves a model on EMNIST plus the Chars74K training split.
void TrainCombined()
{
    var emnist = new EmnistDataset(
        "data/emnist/emnist-byclass-train-images-idx3-ubyte",
        "data/emnist/emnist-byclass-train-labels-idx1-ubyte"
    );

    var emnistSubset = new DatasetSubset( emnist, Enumerable.Range(0, 50000) );

    var chars74k = new Chars74kDataset(
        "data/chars74k/English/Img/GoodImg/Bmp"
    );

    var (charsTrain, _) = DatasetSplitter.StratifiedSplit(chars74k, trainRatio: 0.8, seed: 42);

    var combinedTrain = new CombinedDataset( emnistSubset, charsTrain );

    Console.WriteLine( $"Combined training samples: {combinedTrain.Count}");

    using var model = new CharacterRecognitionCNN();

    var trainer = new Trainer( model, combinedTrain);

    trainer.Train( epochs: 7, batchSize: 64, maxSamples: null );

    Directory.CreateDirectory("models");

    model.save("models/combinedCharRecog_cnn.dat");

    Console.WriteLine("Combined model saved to models/combined_cnn.dat");

}


// Evaluates the combined model on each source dataset separately.
void EvaluateCombined()
{
    const string modelPath = "models/combinedCharRecog_cnn.dat";

    if (!File.Exists(modelPath))
    {
        Console.WriteLine("Combined model not found.");
        return;
    }

    using var model = new CharacterRecognitionCNN();

    model.load(modelPath);


    var emnistTest = new EmnistDataset(
        "data/emnist/emnist-byclass-test-images-idx3-ubyte",
        "data/emnist/emnist-byclass-test-labels-idx1-ubyte"
    );

    Console.WriteLine();
    Console.WriteLine("EMNIST test:");

    var emnistEvaluator = new Evaluator(model, emnistTest);

    emnistEvaluator.Evaluate(batchSize: 64, maxSamples: null);

    var chars74k = new Chars74kDataset(
        "data/chars74k/English/Img/GoodImg/Bmp"
    );

    var (_, charsTest) = DatasetSplitter.StratifiedSplit(chars74k, trainRatio: 0.8, seed: 42);

    Console.WriteLine();
    Console.WriteLine("Chars74K test:");

    var charsEvaluator = new Evaluator(model, charsTest);

    charsEvaluator.Evaluate( batchSize: 64, maxSamples: null);
}

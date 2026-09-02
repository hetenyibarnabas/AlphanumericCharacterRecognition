using AlphanumericCharacterRecognition.Data;
using AlphanumericCharacterRecognition.Model;
using AlphanumericCharacterRecognition.Training;
using AlphanumericCharacterRecognition.Evaluation;

const string modelPath = "models/characterRecognition_cnn.dat";

var trainDataset = new EmnistDataset(
    "data/emnist/emnist-byclass-train-images-idx3-ubyte",
    "data/emnist/emnist-byclass-train-labels-idx1-ubyte"
);

var testDataset = new EmnistDataset(
    "data/emnist/emnist-byclass-test-images-idx3-ubyte",
    "data/emnist/emnist-byclass-test-labels-idx1-ubyte"
);

using var model = new CharacterRecognitionCNN();

if (File.Exists(modelPath))
{
    Console.WriteLine("Loading saved model...");

    model.load(modelPath);

    Console.WriteLine("Model loaded.");
}
else
{
    Console.WriteLine("No saved model found. Starting training...");

    var trainer = new Trainer(model, trainDataset);

    trainer.Train(
        epochs: 1,
        batchSize: 64,
        maxSamples: 10000
    );

    Directory.CreateDirectory("models");

    model.save(modelPath);

    Console.WriteLine($"Model saved to {modelPath}");
}

var evaluator = new Evaluator(model, testDataset);

evaluator.Evaluate(
    batchSize: 64,
    maxSamples: 5000
);
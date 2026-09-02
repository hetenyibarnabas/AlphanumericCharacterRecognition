using AlphanumericCharacterRecognition.Data;
using AlphanumericCharacterRecognition.Model;
using AlphanumericCharacterRecognition.Training;
using AlphanumericCharacterRecognition.Evaluation;

var trainDataset = new EmnistDataset(
    "Data/emnist/emnist-byclass-train-images-idx3-ubyte",
    "Data/emnist/emnist-byclass-train-labels-idx1-ubyte"
);

var testDataset = new EmnistDataset(
    "Data/emnist/emnist-byclass-test-images-idx3-ubyte",
    "Data/emnist/emnist-byclass-test-labels-idx1-ubyte"
);

using var model = new CharacterRecognitionCNN();

var trainer = new Trainer(model, trainDataset);

trainer.Train(
    epochs: 1,
    batchSize: 64,
    maxSamples: 10000
);

var evaluator = new Evaluator(model, testDataset);

evaluator.Evaluate(
    batchSize: 64,
    maxSamples: 5000
);
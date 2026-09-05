using TorchSharp;
using AlphanumericCharacterRecognition.Data;
using AlphanumericCharacterRecognition.Model;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace AlphanumericCharacterRecognition.Evaluation;

/// <summary>
/// Evaluates a trained model on a dataset.
/// </summary>
public class Evaluator
{
    private readonly CharacterRecognitionCNN model;
    private readonly ICharacterDataset dataset;

    /// <summary>
    /// Prepares the model and dataset for evaluation.
    /// </summary>
    public Evaluator( CharacterRecognitionCNN model, ICharacterDataset dataset)
    {
        this.model = model;
        this.dataset = dataset;
    }

    /// <summary>
    /// Computes loss and accuracy on the selected samples.
    /// </summary>
    public void Evaluate(int batchSize = 64,int? maxSamples = null)
    {
        // Evaluation mode selects inference behavior for layers that need it.
        model.eval();

        // Keep the same objective as training so loss values are comparable.
        using var lossFunction = CrossEntropyLoss();

        int sampleCount = maxSamples.HasValue ? Math.Min(maxSamples.Value, dataset.Count) : dataset.Count;

        long correct = 0;
        long processed = 0;
        double totalLoss = 0;
        int batchCount = 0;

        for (int start = 0; start < sampleCount; start += batchSize)    //Enumerate through the batches of the dataset
        {
            int currentBatchSize =  Math.Min(batchSize, sampleCount - start);

            var images = new List<Tensor>();
            var labels = new long[currentBatchSize];

            for (int i = 0; i < currentBatchSize; i++)  //Enumerate through the samples in the current batch.
            {
                var sample = dataset.GetItem(start + i);

                images.Add(sample.Image);
                labels[i] = sample.Label;
            }

            // Match the training batch shape: [N, 3, 32, 32].
            using var inputBatch = torch.stack(images.ToArray());

            // Int64 class indices are the target format for CrossEntropyLoss.
            using var labelBatch = torch.tensor(labels, dtype: ScalarType.Int64);

            // Evaluation runs forward only; no backward pass or optimizer step follows.
            using var predictions = model.forward(inputBatch);

            using var loss = lossFunction.call(predictions, labelBatch);

            // Accuracy uses the class with the highest logit for each image.
            using var predictedLabels = predictions.argmax(1);

            using var correctTensor = predictedLabels.eq(labelBatch).sum();

            correct += correctTensor.ToInt64();

            processed += currentBatchSize;

            totalLoss += loss.item<float>();
            batchCount++;

            foreach (var image in images)
                image.Dispose();
        }

        Console.WriteLine();
        Console.WriteLine("Evaluation results:");
        Console.WriteLine($"Average loss: {totalLoss / batchCount:F4}");
        Console.WriteLine($"Accuracy: {(double)correct / processed:P2}");
    }
}

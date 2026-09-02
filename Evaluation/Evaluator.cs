using TorchSharp;
using AlphanumericCharacterRecognition.Data;
using AlphanumericCharacterRecognition.Model;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace AlphanumericCharacterRecognition.Evaluation;

public class Evaluator
{
    private readonly CharacterRecognitionCNN model;
    private readonly EmnistDataset dataset;

    public Evaluator( CharacterRecognitionCNN model, EmnistDataset dataset)
    {
        this.model = model;
        this.dataset = dataset;
    }

    public void Evaluate(int batchSize = 64,int? maxSamples = null)
    {
        model.eval();

        using var lossFunction = CrossEntropyLoss();

        int sampleCount = maxSamples.HasValue ? Math.Min(maxSamples.Value, dataset.Count) : dataset.Count;

        long correct = 0;
        long processed = 0;
        double totalLoss = 0;
        int batchCount = 0;

        for (int start = 0; start < sampleCount; start += batchSize)
        {
            int currentBatchSize =  Math.Min(batchSize, sampleCount - start);

            var images = new List<Tensor>();
            var labels = new long[currentBatchSize];

            for (int i = 0; i < currentBatchSize; i++)
            {
                var sample = dataset.GetItem(start + i);

                images.Add(sample.Image);
                labels[i] = sample.Label;
            }

            using var inputBatch = torch.stack(images.ToArray());

            using var labelBatch = torch.tensor(labels, dtype: ScalarType.Int64);

            using var predictions = model.forward(inputBatch);

            using var loss = lossFunction.call(predictions, labelBatch);

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
        Console.WriteLine(
            $"Average loss: {totalLoss / batchCount:F4}"
        );
        Console.WriteLine(
            $"Accuracy: {(double)correct / processed:P2}"
        );
    }
}
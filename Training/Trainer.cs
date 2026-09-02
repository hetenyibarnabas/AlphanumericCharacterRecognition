using TorchSharp;
using AlphanumericCharacterRecognition.Data;
using AlphanumericCharacterRecognition.Model;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace AlphanumericCharacterRecognition.Training;

/// <summary>
/// Trains the character recognition model on EMNIST samples.
/// </summary>
public class Trainer
{
    private readonly CharacterRecognitionCNN model;
    private readonly ICharacterDataset dataset;

    /// <summary>
    /// Prepares the model and training dataset for the training loop.
    /// </summary>
    public Trainer(CharacterRecognitionCNN model, ICharacterDataset dataset)
    {
        this.model = model;
        this.dataset = dataset;
    }

    /// <summary>
    /// Trains the model in mini-batches while tracking loss and accuracy.
    /// </summary>
    public void Train( int epochs = 1, int batchSize = 64, int? maxSamples = null)
    {
        model.train();

        // Adam adaptively updates the CNN weights from mini-batch gradients.
        using var optimizer = torch.optim.Adam( model.parameters(), lr: 0.001);

        // CrossEntropyLoss expects raw logits [N, classes] and Int64 labels [N].
        using var lossFunction = CrossEntropyLoss();

        int sampleCount = maxSamples.HasValue ? Math.Min(maxSamples.Value, dataset.Count) : dataset.Count;

        int[] indices = Enumerable.Range(0, sampleCount).ToArray();

        for (int epoch = 1; epoch <= epochs; epoch++)
        {
            // Shuffle each epoch to avoid learning from a fixed sample order.
            Shuffle(indices);

            double totalLoss = 0;
            long correct = 0;
            long processed = 0;
            int batchNumber = 0;

            for (int start = 0; start < sampleCount; start += batchSize)
            {
                int currentBatchSize = Math.Min(batchSize, sampleCount - start);

                var imageList = new List<Tensor>();
                var labelValues = new long[currentBatchSize];

                for (int i = 0; i < currentBatchSize; i++)
                {
                    int datasetIndex = indices[start + i];

                    var sample = dataset.GetItem(datasetIndex);

                    imageList.Add(sample.Image);
                    labelValues[i] = sample.Label;
                }

                // Stack samples into the batch shape consumed by Conv2d: [N, 3, 32, 32].
                using var inputBatch = torch.stack(imageList.ToArray());

                // Class labels stay as a 1D Int64 tensor for CrossEntropyLoss.
                using var labelBatch = torch.tensor(labelValues, dtype: ScalarType.Int64);

                // Gradients accumulate by default, so clear them before this batch.
                optimizer.zero_grad();

                // Forward pass produces one logit vector per image.
                using var predictions = model.forward(inputBatch);

                using var loss = lossFunction.call(predictions, labelBatch);

                // Backpropagation fills parameter gradients from the loss.
                loss.backward();

                // Apply Adam's parameter update using the gradients just computed.
                optimizer.step();

                totalLoss += loss.item<float>();

                // The largest logit along the class dimension is the predicted label.
                using var predictedLabels = predictions.argmax(1);

                using var correctTensor = predictedLabels.eq(labelBatch).sum();

                correct += correctTensor.ToInt64();
                processed += currentBatchSize;

                foreach (var image in imageList)  image.Dispose();

                batchNumber++;

                if (batchNumber % 50 == 0)
                {
                    Console.WriteLine(
                        $"Epoch {epoch} | " +
                        $"{processed}/{sampleCount} | " +
                        $"Loss: {loss.item<float>():F4} | " +
                        $"Accuracy: {(double)correct / processed:P2}"
                    );
                }
            }

            Console.WriteLine();
            Console.WriteLine(
                $"Epoch {epoch} finished | " +
                $"Average loss: {totalLoss / batchNumber:F4} | " +
                $"Accuracy: {(double)correct / processed:P2}"
            );
        }
    }

    /// <summary>
    /// Randomizes sample indices before an epoch.
    /// </summary>
    private static void Shuffle(int[] array)
    {
        for (int i = array.Length - 1; i > 0; i--)
        {
            int j = Random.Shared.Next(i + 1);

            (array[i], array[j]) =  (array[j], array[i]);
        }
    }
}

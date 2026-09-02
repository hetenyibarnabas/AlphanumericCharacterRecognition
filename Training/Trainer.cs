using TorchSharp;
using AlphanumericCharacterRecognition.Data;
using AlphanumericCharacterRecognition.Model;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace AlphanumericCharacterRecognition.Training;

public class Trainer
{
    private readonly CharacterRecognitionCNN model;
    private readonly EmnistDataset dataset;

    public Trainer(CharacterRecognitionCNN model, EmnistDataset dataset)
    {
        this.model = model;
        this.dataset = dataset;
    }

    public void Train( int epochs = 1, int batchSize = 64, int? maxSamples = null)
    {
        model.train();

        using var optimizer = torch.optim.Adam( model.parameters(), lr: 0.001);

        using var lossFunction = CrossEntropyLoss();

        int sampleCount = maxSamples.HasValue ? Math.Min(maxSamples.Value, dataset.Count) : dataset.Count;

        int[] indices = Enumerable.Range(0, sampleCount).ToArray();

        for (int epoch = 1; epoch <= epochs; epoch++)
        {
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

                using var inputBatch = torch.stack(imageList.ToArray());

                using var labelBatch = torch.tensor(labelValues, dtype: ScalarType.Int64);

                optimizer.zero_grad();

                using var predictions = model.forward(inputBatch);

                using var loss = lossFunction.call(predictions, labelBatch);

                loss.backward();

                optimizer.step();

                totalLoss += loss.item<float>();

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

    private static void Shuffle(int[] array)
    {
        for (int i = array.Length - 1; i > 0; i--)
        {
            int j = Random.Shared.Next(i + 1);

            (array[i], array[j]) =  (array[j], array[i]);
        }
    }
}
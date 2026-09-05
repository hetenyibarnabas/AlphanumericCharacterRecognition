# Alphanumeric Character Recognition

A convolutional neural network for recognizing written and printed alphanumeric characters.
The project is implemented in **C#** using **TorchSharp**.

## Supported characters
- `0-9`, `A-Z`, `a-z`

## Datasets

- **EMNIST ByClass**; handwritten characters.
- **Chars74K**; printed characters.

Dataset files aren't in this repo.

## Model

Input: 3 x 32 x 32 RGB image

```text
Conv2D 3 → 32
ReLU
MaxPool
Conv2D 32 → 64
ReLU
MaxPool
Linear 4096 → 128
ReLU
Linear 128 → 62
```

## Results

EMNIST ByClass: 82.43%
Chars74K: 70.54%

## Usage

Restore dependencies:
```bash
dotnet restore
```

Train the combined model:
```bash
dotnet run -- train-combined
```

Evaluate the combined model:
```bash
dotnet run -- evaluate-combined
```

## Technologies
- C#, .NET, TorchSharp, SkiaSharp

## License
MIT

(This CNN is built built for learning purposes after I read about Yann Le Cun's role and labor in the field of image recognition and computer vision.)

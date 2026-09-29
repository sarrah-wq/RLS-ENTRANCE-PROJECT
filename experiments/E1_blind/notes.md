# E1 — Blind Baseline

## Question

Does the model use the image?

## Hypothesis

I expect the model to perform much worse when the images are replaced by zeros, because it loses the visual information needed to identify size, color, shape, and spatial relations.

## Baseline

The normal model was trained with the original ShapeScenes images.

Configuration:
- seed: 42
- d_model: 128
- n_heads: 4
- n_layers: 2
- batch size: 64
- learning rate: 0.001
- epochs: 15

Normal test results:
- Exact match: 70.35%
- Size accuracy: 84.14%
- Color accuracy: 74.40%
- Shape accuracy: 74.24%
- Relation accuracy: 43.43%
- Letter accuracy: 69.76%

## One variable changed

For the blind baseline, the same model and training setup were used, but every image was replaced with a tensor of zeros during training and validation.

## Blind results

- Exact match: 1.65%
- Size accuracy: 33.28%
- Color accuracy: 20.10%
- Shape accuracy: 20.63%
- Relation accuracy: 0.00%
- Letter accuracy: 17.05%
- Unparsed predictions: 0 / 2000

## Comparison

| Metric | Normal | Blind |
|---|---:|---:|
| Exact match | 70.35% | 1.65% |
| Size | 84.14% | 33.28% |
| Color | 74.40% | 20.10% |
| Shape | 74.24% | 20.63% |
| Relation | 43.43% | 0.00% |
| Letter | 69.76% | 17.05% |

## Interpretation

Replacing the images with zeros reduced exact-match accuracy from 70.35% to 1.65%. Accuracy also decreased for size, color, shape, relation, and letters. This supports the conclusion that the model's normal performance depends substantially on visual information.

The blind model still achieved some non-zero accuracy, which suggests that it can learn some structure from the character sequences and dataset grammar even without useful visual input.
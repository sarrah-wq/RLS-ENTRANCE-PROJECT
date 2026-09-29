# AI Usage

AI tools were used during development of this project.

## How AI was used

ChatGPT was used to:

- explain PyTorch, CNN, Transformer, attention, masking, teacher forcing, and training concepts;
- help debug Python and PyTorch errors;
- review code structure and tensor shapes;
- compare the implementation with the RLS project requirements;
- explain experimental results.

## What I did myself

I ran the code and experiments locally in my own VS Code environment.
implement simple evaluation and benchmarking scripts
I checked the implementation by running the provided tests and project scripts.

I trained the normal and blind models, ran the evaluation on the test and held-out splits, and ran the S1 throughput benchmark.

The reported results in the repository come from my own runs.

## Verification

The provided test suite was run successfully:

    100 passed, 1 skipped

The skipped test was related to CUDA because CUDA was not available in the current PyTorch installation.

I reviewed and tested the generated code before using it in the project.
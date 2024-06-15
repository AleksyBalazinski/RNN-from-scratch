# RNNFromScratch
Recursive neural network for handwritten digit recognition.
Uses a small custom autodiff library.

## Usage
Run with
```
> julia src/RNNFromScratch.jl
```
To enable profiling:
```
> julia src/RNNFromScratch.jl withprof
```

## Paper
Abstract:
>This paper presents a small deep learning library in Julia, showcasing its expressiveness and competitive performance.
Our implementation achieved 93\% accuracy after five epochs, improving to 95\% with ten for the MNIST handwritten digit recognition problem.
Comparative analysis with Flux and TensorFlow showed similar accuracy and efficiency, validating Julia's potential for machine learning.
The library's user-friendly interface highlights Julia as a promising option for deep learning development.

Overleaf link: https://www.overleaf.com/read/wsmmctxqckwf#983966
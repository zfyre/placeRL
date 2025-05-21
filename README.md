# PlaceRL: Fast Chip Placement via Reinforced  Visual Representation Learning

A new chip placement method based on visual representation learning. 

## Experimental Report

[Adaptec-1 and Ariane Experimental Results using DDPG and PPO ](https://wandb.ai/zfyre-iit-roorkee/placerl/reports/Experimental-Results---VmlldzoxMjU3NjQwMg)

## Usage

You can start easily by using the following script.

```
cd maskplace
python PPO2.py
```

## Parameter

- **gamma** Decay factor.
- **seed** Random seed.
- **disable_tqdm** Whether to disable the progress bar.
- **lr** Learning rate.
- **log-interval** Interval between training status logs.
- **pnm** Number of place modules for each placement trajectory. 
- **benchmark** Circuit benchmark.
- **soft_coefficient** Whether to constriant the actions based on the wiremask.
- **batch_size** Batch size.
- **is_test** Testing mode based on the trained agent.
- **save_fig** Whether to save placement figures.


## Benchmark
The repo has provided the benchmark *adaptec1* and *ariane*. For other benchmarks, you can download them by the following the link:

[click here](http://www.cerc.utexas.edu/~zixuan/ispd2005dp.tar.xz)


## Dependency
- [Python](https://www.python.org/) >= 3.9

- [Pytorch](https://pytorch.org/) >= 1.10

  - Other versions may also work, but not tested

- [gym](https://www.gymlibrary.dev/index.html) >= 0.21.0
- [matplotlib](https://matplotlib.org/) >= 3.7.1
- [tqdm](https://tqdm.github.io/)
- [protobuf](https://pypi.org/project/protobuf/) (for benchmark *ariane* (protobuf==3.20))

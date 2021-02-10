# annotated-ml-models
A collection of implementations of popular ML models with explanatory annotations.

Most models are implemented in PyTorch.

PRs are welcome :)

## Models
- [yolo-v3](yolo-v3/)

## Download & Installation
Clone the repository with:
```git clone https://github.com/davidmaamoaix/annotated-ml-models.git```

Remember to `cd` into the directory:
```cd annotated-ml-models```

## Usage
Each model has its subdirectory containing the following files:
- `<model_name>.py`: the model's code
- `train_test.py`: the script for training & testing the model

The `train_test.py` script can also be invoked from the command line:

```python <model_directory_name>/train_test.py train```

for training, and replace `train` with `test` for testing. If the associated dataset has not been downloaded, the script will automatically download it.

The parameters of the model is saved in its directory after training.
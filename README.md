# Installing PyTorch, TensorFlow, and JAX on Apple Silicon

Create a virtual environment and update pip:

```
$ python3.12 -m venv .venv
```

Install TensorFlow:

```
$ source .venv/bin/activate && pip install --upgrade pip && pip install tensorflow tensorflow-macos tensorflow-metal
```

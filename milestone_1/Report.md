# Milestone 1: Group Marxer, Kotatis, Rohrer
This report documents our progress for the first milestone.
## Task 1:
The dataset assigned to us was constructed for a sentiment analysis publication from a collection of reviews from IMDB. The authors Maas et al. released this dataset to the public, it can be found online ([https://ai.stanford.edu/~amaas/data/sentiment/](https://ai.stanford.edu/~amaas/data/sentiment/)) and in their paper ([https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf](https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf)). Furthermore, the dataset can be imported using the Keras package in Python, which we will demonstrate later on.

The dataset is about movie reviews and their binary sentiment polarity. It contains in total 50’000 reviews. Half of them are in the training set and the other half in the test set. The overall distribution of positive and negative labeled reviews is evenly distributed 25’000 positive and 25’000 negative. Furthermore, there are only 30 reviews allowed for each movie, because reviews for the same movie tend to have correlated ratings.

Reviews with neutral ratings are not included in the train- and test set. The sets contain only clear labeled positive (score >= 7) and negative (score <= 4) ratings.

The problem solved by the machine learning models is a classical one and the aim is to distinguish whether a certain movie review has a positive or negative meaning.

## Task 2:
For the second task we check out the code base assigned to us from GitHub.
The code is listed in the keras repository of the Keras-Team on GitHub. The team consists of six users, while our file was put together by another eight contributors. Our code is called imdb_cnn.py, which indicates a Python syntax. It is saved in the examples folder of the repository.

It includes helpful comments and begins by promting the import of a few objects, namely Keras layers and a dataset. The latter is identical to our dataset from task one, which has evidently been built-in into the Keras API. Documentation is provided by Keras ([https://keras.io/api/datasets/imdb/](https://keras.io/api/datasets/imdb/)). This integration proves helpful for our further tasks.

The code uses this dataset to demonstrate the use of Convolution1D for text classification. Documentation for the 1D convolution layer can found unter ([https://keras.io/api/layers/convolution_layers/convolution1d/](https://keras.io/api/layers/convolution_layers/convolution1d/)).

## Task 3:
To commit the relevant Python file from the second task to our Git repository, we configure our terminals with our GitHub accounts and the repository that we set up for this course ([https://github.com/Stringz/data-science-toolkits-and-architectures](https://github.com/Stringz/data-science-toolkits-and-architectures)). 
We clone the repository to our local machines with `$ git clone URL`. We then save a simple text file in the local directory that was created when we cloned the repository. Using `$ git add .` we can stage this file for commit to our local repository, and with `$ git commit -m "commit description"`we can commit the staged file. Lastly, we use `$ git push origin` to push the file to GitHub. Next, we can make changes to this file or add new files for further commits. Every commit should include a meaningful comment, that summarizes the changes and gives a quick insight about why that was necessary.
## Task 4:
To run the python code on our Linux system, we first need to install pip via the terminal. Therefore we use the command `sudo su` first, to afterwards use the command `apt install python3-pip as root`. After that we installed the necessary packages for our code, namely Tensorflow and Keras, but the latter comes packaged with TensorFlow 2.0. Use the command `pip install tensorflow`.
Later on we save the code as a python file, by copying it from the website and using a simple text editor. Lastly we change the working directory in the terminal to the directory where we saved the code file. We do this with the help of the command cd. Finally we are able to run the code with the command `python3 Filename.py`.

With the help of the command `python3 - V`, we find out that we use Version 3.8.5 of Python. We use the command `pip3 list` to find out the versions of the packages. We use Version 2.3.1 of Tensorflow and Version 2.4.3 of Keras.

We try to run this code on different systems. It works just as well with Python 3.8.3 on a macintosh terminal. To test it out of the box, we try to run it on the provided server. We try to use a requirements file to install Tensorflow on the linux server.
## Task 5:
The input to the neural networks are different movie reviews and the rating which goes in hand with a certain review. The output contains different indicators. The first one is the variable loss = 0.2279. The loss is a prediction error of the neural network. The second output is the accuracy = 0.9088 which is the fraction of predictions that the model got right. The last two output values, val_accuracy and val_loss are similar to loss and accuracy, but only based during the validation test with the testing set. These two indicators are used to evaluate whether there is a potential problem of overfitting or not.

Keras is a deep learning library written in python. Keras provides several API's for different backends including Tensorflow. Even though Keras is part of the Tensorflow Core API, it is still an independent library.

The data is loaded with the help of the keras.datasets package. This package contains several datasets that can be directly loaded. Therefore we use the command `imdb.load_data`.

We install the package pipdeptree to find out which dependencies come in hand with the imported modules. We then run piptree and search all the dependencies of our imported modules Tensorflow and Keras, which are listed below in the Appendix.

To finish of this task with the last question, the architecture of the neural network is a feed-forward neural network. The first layer is the input and the last one the Output.
## Task 6:
For task 6 we need to create a documentation file for our code, and add it to the root folder of our GitHub project. This file should explain step by step how one can run the code, out of the box.

## Additional implementation:
Plot results using matplotlib. Install matplotlib with `pip install matplotlib`.

    import matplotlib.pyplot as plt
    plt.plot(fit.history["val_loss"])
    plt.title("Validation Loss History")
    plt.ylabel("Loss Value")
    plt.xlabel("# Epochs")
    plt.show()
    plt.plot(fit.history["val_accuracy"])
    plt.title("Validation Accuracy History")
    plt.ylabel("Accuracy Value (%)")
    plt.xlabel("# Epochs")
    plt.show()

## Appendix:
List of dependencies from Task 5:
Keras==2.4.3

- h5py [required: Any, installed: 2.10.0]

- numpy [required: >=1.7, installed: 1.18.5]

- six [required: Any, installed: 1.14.0]

- numpy [required: >=1.9.1, installed: 1.18.5]

- pyyaml [required: Any, installed: 5.3.1]

- scipy [required: >=0.14, installed: 1.5.2]

- numpy [required: >=1.14.5, installed: 1.18.5]

  

Tensorflow==2.3.1

- absl-py [required: >=0.7.0, installed: 0.10.0]

- six [required: Any, installed: 1.14.0]

- astunparse [required: ==1.6.3, installed: 1.6.3]

- six [required: >=1.6.1,<2.0, installed: 1.14.0]

- wheel [required: >=0.23.0,<1.0, installed: 0.34.2]

- gast [required: ==0.3.3, installed: 0.3.3]

- google-pasta [required: >=0.1.8, installed: 0.2.0]

- six [required: Any, installed: 1.14.0]

- grpcio [required: >=1.8.6, installed: 1.32.0]

- six [required: >=1.5.2, installed: 1.14.0]

- h5py [required: >=2.10.0,<2.11.0, installed: 2.10.0]

- numpy [required: >=1.7, installed: 1.18.5]

- six [required: Any, installed: 1.14.0]

- keras-preprocessing [required: >=1.1.1,<1.2, installed: 1.1.2]

- numpy [required: >=1.9.1, installed: 1.18.5]

- six [required: >=1.9.0, installed: 1.14.0]

- numpy [required: >=1.16.0,<1.19.0, installed: 1.18.5]

- opt-einsum [required: >=2.3.2, installed: 3.3.0]

- numpy [required: >=1.7, installed: 1.18.5]

- protobuf [required: >=3.9.2, installed: 3.13.0]

- setuptools [required: Any, installed: 45.2.0]

- six [required: >=1.9, installed: 1.14.0]

- six [required: >=1.12.0, installed: 1.14.0]

- tensorboard [required: >=2.3.0,<3, installed: 2.3.0]

- absl-py [required: >=0.4, installed: 0.10.0]

- six [required: Any, installed: 1.14.0]

- google-auth [required: >=1.6.3,<2, installed: 1.22.1]

- cachetools [required: >=2.0.0,<5.0, installed: 4.1.1]

- pyasn1-modules [required: >=0.2.1, installed: 0.2.8]

- pyasn1 [required: >=0.4.6,<0.5.0, installed: 0.4.8]

- rsa [required: >=3.1.4,<5, installed: 4.6]

- pyasn1 [required: >=0.1.3, installed: 0.4.8]

- setuptools [required: >=40.3.0, installed: 45.2.0]

- six [required: >=1.9.0, installed: 1.14.0]

- google-auth-oauthlib [required: >=0.4.1,<0.5, installed: 0.4.1]

- google-auth [required: Any, installed: 1.22.1]

- cachetools [required: >=2.0.0,<5.0, installed: 4.1.1]

- pyasn1-modules [required: >=0.2.1, installed: 0.2.8]

- pyasn1 [required: >=0.4.6,<0.5.0, installed: 0.4.8]

- rsa [required: >=3.1.4,<5, installed: 4.6]

- pyasn1 [required: >=0.1.3, installed: 0.4.8]

- setuptools [required: >=40.3.0, installed: 45.2.0]

- six [required: >=1.9.0, installed: 1.14.0]

- requests-oauthlib [required: >=0.7.0, installed: 1.3.0]

- oauthlib [required: >=3.0.0, installed: 3.1.0]

- requests [required: >=2.0.0, installed: 2.22.0]

- grpcio [required: >=1.24.3, installed: 1.32.0]

- six [required: >=1.5.2, installed: 1.14.0]

- markdown [required: >=2.6.8, installed: 3.3.1]

- numpy [required: >=1.12.0, installed: 1.18.5]

- protobuf [required: >=3.6.0, installed: 3.13.0]

- setuptools [required: Any, installed: 45.2.0]

- six [required: >=1.9, installed: 1.14.0]

- requests [required: >=2.21.0,<3, installed: 2.22.0]

- setuptools [required: >=41.0.0, installed: 45.2.0]

- six [required: >=1.10.0, installed: 1.14.0]

- tensorboard-plugin-wit [required: >=1.6.0, installed: 1.7.0]

- werkzeug [required: >=0.11.15, installed: 1.0.1]

- wheel [required: >=0.26, installed: 0.34.2]

- tensorflow-estimator [required: >=2.3.0,<2.4.0, installed: 2.3.0]

- termcolor [required: >=1.1.0, installed: 1.1.0]
- wheel [required: >=0.26, installed: 0.34.2]
- wrapt [required: >=1.11.1, installed: 1.12.1]

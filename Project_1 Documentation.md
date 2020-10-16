# Project_1 Documentation
Follow these steps to run the code on the environment of your choice.



# Before running the code

To run the Python file, start your Python console and make sure your version is up to date. Use version 3.8.3 or higher to avoid any potential troubles.
To install the necessary packages, our codes uses the pip-function, which links to the Python Packaging Index repository PyPI and ensures a good installation.
If you don't have pip yet, install it using the following code:

    sudo apt install python3-pip

Non-root users may need to set up a virtual environment. 
With the pip function ready, we can use it to download Keras and Tensorflow:

    pip install keras
    pip install tensorflow

Note that Keras and Tensorflow may also load some dependent components, dependencies, if they aren't found on your system. Keras itself is one of the dependencies of Tensorflow 2.0, so the command `pip install tensorflow` should be sufficient. 
Lastly, you need to install MatPlotLib with `pip install matplotlib`.

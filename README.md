# Tutorials
Tutorials for Special Topics in Advanced Machine Learning class 

# Description
Content:
* Readme.md - this file
* adversarial - demonstrate GAN on mnist
* autoencoder - demonstrate autoencoders on mnist
* classification - demonstrate classification on mnist
* regression - demonstrate simple regression (runs on CPU)

Each tutorial folder contains:
* doall.lua - main script, which executes other files
* loadMnist.lua - downloading and loading mnist dataset
* model.lua - creates NN model
* saveModel.lua - sets of functions to to remove temporary data from model and save it 
* train.lua - sets of functions for training NN and main training loop

# Running
From a selected tutorial directory run:
```
qlua doall.lua
```

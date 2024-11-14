# DeepAerofoil
This project was my tool to apply the Basic Principles of Deep Learning in a topic I appreciate -- Airfoils. Two novel Neural Networks were created in PyTorch (regression model and convolutional neural network) that will predict the Cl, Cd, and Cm of the airfoil based on its shape and other specified conditions like the angle of attack and Reynolds number. Also the results were compared to a similar project and proved to be more efficient and use extensively less computational time and resources. [1](https://github.com/vicmcl/deeplearwing/tree/main)

This project involved compiling a comprehensive dataset of 110 unique airfoils, focusing on essential aerodynamic parameters such as:
* Reynolds number (Re)
* Lift coefficient (Cl)
* Drag coefficient (Cd)
* Moment coefficient (Cm)
* Angle of Attack (AoA)

### Whats New?
The original study plotted the image of airfoils then fed it into the model which required both extensive amount of resources. What this project offers is Simplicity. Only represented points instead of the picture which is only plotted within the model and got more accurate results in terms of Absloute Mean Error for the predicted Cl, Cd, and Cm. It's often deceiving to go for the whole resources and overlooking the easier way to do it. In this project the same procedure was undertaked and much easier route too. Both Models predicted more accurae results. The sample was less though. Moreever, the CPU used to chrash under 16 GB of RAM, now it will work effeciently with only 2 GB RAM.

### Overview / Steps
The general steps were followed as any other ML project:
1. Collect and Prepare the Data.
2. Data-Preprocessing.
3. Build the training model.
4. Fit the data into the model and make predictions
5. Evaluate and Improve the model.
The project also employed scikit-learn for data splitting, normalization, and evaluation of the model.

### Key Features

Data Compilation: Gathered a robust dataset of 110 distinct airfoils, each with aerodynamic characteristics measured for varying Reynolds numbers.
Model Architecture: Constructed a CNN model capable of analyzing airfoil shapes and predicting the lift and drag coefficients.
Training Process: Leveraged PyTorch for model training with optimizers and learning rate scheduling to ensure efficient convergence.
Data Handling: Utilized scikit-learn tools for:
Data splitting (training and testing sets)
Data normalization to enhance model performance
Model evaluation with appropriate metrics
Performance Validation: Assessed the trained model using metrics such as:
Mean Squared Error (MSE)
R-squared (R²) score for determining the model's accuracy and reliability.

## Datasets
Three datasets in the data file were used.
### Regression Model
For this model hte only acceptable data is numbers so the geometry of each airfoil were normalized and represented by 12 values in the upper and lower surfaces. For the methodolgy I followed, refer to this paper. [2](https://github.com/Mohamedelrefaie/TransonicSurrogate/tree/main) This was under the file xfoil_Reg. (Link to the dataset)

### Convolutional Neural Network
The Cl, Cd, and Cm were extracted from XFOIL at first then just complemented with data on [Airfoils Tools](http://airfoiltools.com/) as they appeared to be the same. and the file is presented as "xfoil_data.csv". (A more extensive list can be found here.)
In the first model, the coordinates were plotted in an Excel sheet and 20 points were chosen at 0-->1 taking steps of 0.1 to represent the Airfoil.
See Pic {picture link here./}
In the second model, the whole points were used to plot the shape of the airfoil and then feeds it into the model.

Comprehensive dataset of 110 unique airfoils
Key parameters collected:

Reynolds number
Lift coefficient
Drag coefficient
Moment coefficient


Data sourced from XFOIL simulations
Available in xfoil_data.csv


## Model Architectures
Model 1: Regression

Code of the neural network on regression example and maybe a representaion.

Model 2: CNN

Uses complete airfoil coordinate sets
Direct shape processing through convolutional layers
Enhanced feature extraction capabilities


## Methodology
The project follows a structured machine learning workflow:

1. Data Collection and Preparation

Gathering airfoil coordinates
Extracting aerodynamic coefficients from XFOIL
Dataset compilation and organization


2. Data Preprocessing

Feature scaling and normalization using scikit-learn
Data splitting into training and validation sets
Input formatting for respective models


3. Model Development

PyTorch implementation
CNN architecture design
Training pipeline setup


Training and Prediction

Model training with prepared dataset
Hyperparameter optimization
Prediction generation


Model Evaluation

Performance metrics:

Mean Squared Error (MSE)
R-squared (R²) scores


Model validation and testing
Iterative improvement


## Model Performance
Compared to the original study by TensorFlow, the results for both Regression and Mathimatical model not only exceeds them, they are much faster.
| MAE | Original Study | Current Results |
| :---: | :---: | :---: |
| Cl | 0.1081 | 0.1023 |
| Cd | 0.0106 | 0.0076 |
| Cm | 0.0155 | 0.0143 |

One Other important factor to consider is the avilability of resources: RAM GB and Time invested.
Validation performance metrics available in training logs


## Resources and Refrences
* [1] DeepLearWing. [link](https://github.com/vicmcl/deeplearwing/tree/main)
* [2] Surrogate Modeling of the Aerodynamic Performance for Transonic Regime. [link](https://github.com/Mohamedelrefaie/TransonicSurrogate/tree/main)

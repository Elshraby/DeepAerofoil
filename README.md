# DeepAerofoil
This project presents a novel machine learning framework for predicting aerodynamic characteristics of airfoils using advanced neural network architectures.It was my tool to apply the Basic Principles of Deep Learning in a topic I appreciate -- Airfoils. Two novel Neural Networks were created in PyTorch (regression model and convolutional neural network) that will predict the Cl, Cd, and Cm of the airfoil based on its shape and other specified conditions like the angle of attack and Reynolds number. Also the results were compared to a similar project and proved to be more accurate and use extensively less computational time and resources. [[1](https://github.com/vicmcl/deeplearwing/tree/main)]

## Introduction
Traditional computational fluid dynamics (CFD) approaches are computationally expensive and time-consuming, motivating the development of machine learning techniques that can provide rapid and accurate predictions. A Promising alternative that is being explored nowadays is surrogate moddeling. A previous study plotted the image of airfoils then fed it into the model; however, this technique required  extensive amount of resources and time too, losing the only benifit provided by using machine learning in the first place.

### Whats New?
This project introduces simplicity as its key innovation. Instead of using full airfoil images, which require extensive computational resources, this approach uses only the represented points of the airfoil. These points are fed into the model, resulting in more accurate predictions of the aerodynamic coefficients Cl, Cd, and Cm in terms of Absolute Mean Error (AME).

It's easy to get caught up in using complex techniques, but sometimes a simpler approach can yield better results. By tweaking the conceptualization of the problem, this project successfully implements a much more efficient method. Despite using a smaller sample size, the model achieved better accuracy, and the computational efficiency was vastly improved. Whereas the previous approach required a system with 16 GB of RAM and struggled to operate, this method now runs smoothly with just 2 GB of RAM.

## 2. Methodology

### 2.1 Data Acquisition and Preprocessing

#### 2.1.1 Dataset Composition
* **Total Airfoil Samples**: 110 unique airfoil geometries
* **Data Sources**:
   * XFOIL computational simulations
   * Airfoil Tools online repository
* **Key Aerodynamic Parameters**:
   * Reynolds Number (Re)
   * Lift Coefficient (Cl)
   * Drag Coefficient (Cd)
   * Moment Coefficient (Cm)
   * Angle of Attack (AoA)

#### 2.1.2 Data Representation Strategies
We developed two distinct data representation approaches to capture airfoil geometries and aerodynamic characteristics:

1. **Regression Model Representation**:
   * Airfoil geometry normalized to 12 representative points
   * Upper and lower surface coordinates mathematically transformed
   * Focused on numerical feature extraction
   * **Advantages**:
     - Reduced computational complexity
     - Simplified feature representation
     - Rapid processing and inference

2. **Convolutional Neural Network (CNN) Representation**:
   * Full coordinate set preservation
   * Two variations explored: 
     a) 20-point sampling (0 → 1, step 0.1)
     b) Complete point cloud representation
   * **Advantages**:
     - Comprehensive geometric information capture
     - Enhanced feature learning capabilities
     - Robust to geometric variations

### 2.2 Neural Network Architectures

#### 2.2.1 Regression Model
* **Architecture**: Fully connected neural network
* **Input**: Normalized geometric features
* **Objective**: Direct coefficient prediction
* **Key Characteristics**:
   * Minimal computational overhead
   * Rapid training and inference
   * Low resource requirements (2 GB RAM)

#### 2.2.2 Convolutional Neural Network (CNN)
* **Architecture**: Convolutional layers with geometric feature extraction
* **Input**: Airfoil coordinate representations
* **Objective**: Comprehensive shape-based coefficient prediction
* **Key Characteristics**:
   * Enhanced feature learning capabilities
   * Deeper geometric information processing
   * Ability to capture complex geometric relationships
 

![](images/RegMatch.png)

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

This project involved compiling a comprehensive dataset of 110 unique airfoils, focusing on essential aerodynamic parameters such as:
* Reynolds number (Re)
* Lift coefficient (Cl)
* Drag coefficient (Cd)
* Moment coefficient (Cm)
* Angle of Attack (AoA)

### Regression Model
For this model hte only acceptable data is numbers so the geometry of each airfoil were normalized and represented by 12 values in the upper and lower surfaces. For the methodolgy I followed, refer to this paper. [2](https://github.com/Mohamedelrefaie/TransonicSurrogate/tree/main) This was under the file xfoil_Reg. (Link to the dataset)
![](images/AirfoilShape.png)
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

![](images/Regression.png)

Model 2: CNN

Uses complete airfoil coordinate sets
Direct shape processing through convolutional layers
Enhanced feature extraction capabilities
![](images/CNNStdDev.png)

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
| MAE | Original Study | Current Cov Results | Current Reg Results |
| :---: | :---: | :---: | :---: |
| Cl | 0.1081 | 0.1017 | 0.0215 |
| Cd | 0.0106 | 0.0078 | 0.0057 |
| Cm | 0.0155 | 0.0154 | 0.0071 |

While it didn't capture the whole geometry of the airfoil, it exceeded the CNN results with only 5 min to train and test.
One Other important factor to consider is the avilability of resources: RAM GB and Time invested.
Validation performance metrics available in training logs

## Acknoledgment
This project was conducted as an independent academic exploration of machine learning techniques in aerospace engineering.

## Resources and Refrences
* [1] DeepLearWing. [link](https://github.com/vicmcl/deeplearwing/tree/main)
* [2] Surrogate Modeling of the Aerodynamic Performance for Transonic Regime. [link](https://github.com/Mohamedelrefaie/TransonicSurrogate/tree/main)

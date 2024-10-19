# DeepAerofoil
This project was jut a tool to apply the Basic Principles of Deep Learning in a topic I appreciate -- Airdoils. Two models were created the first is a regression model and the second is a convolutional neural network. The model will predict the Cl, Cd, and Cm of the airfoil based on the shape of the airfoil and the angle of attack.
The M.O. will be the same as any other Machine Learning Project.
1. Collect and Prepare the Data.
2. Data-Preprocessing.
3. Build the training model.
4. Fit the data into the model and make predictions
5. Evaluate and Improve the model.

## Collect the data.
The Cl, Cd, and Cm were extracted from XFOIL and the file is presented as "xfoil_data.csv". (A more extensive list can be found here.)
In the first model, the coordinates were plotted in an Excel sheet and 20 points were chosen at 0-->1 taking steps of 0.1 to represent the Airfoil.
See Pic {picture link here./}
In the second model, the whole points were used to plot the shape of the airfoil and then feeds it into the model.

## Data-Preprocessing

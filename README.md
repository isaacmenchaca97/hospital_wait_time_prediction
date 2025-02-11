# Hospital Wait Time Prediction

## Overview

This project aims to predict hospital wait times using machine learning techniques. The model is designed to assist hospitals in better managing patient flow and reducing waiting periods based on historical data.

## Dataset

The dataset includes various features related to hospital visits, such as:

- Patient demographics (age, gender, etc.)
- Time-related features (hour, day, month, etc.)
- Patient classification
- Previous wait times

## Model Architecture

The model utilizes regression techniques, including K-Nearest Neighbors, Decision Trees, and a Multilayer Perceptron Neural Network, to predict wait times based on input features.

## Installation & Requirements

To run this project, ensure you have the following installed:

- Python 3.x
- Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `tensorflow==2.15.0`

## Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/isaacmenchaca97/hospital_wait_time_prediction.git
   cd hospital_wait_time_prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the model training script:
   ```bash
   python train_model.py
   ```

## Results

- The trained model provides estimated wait times based on input features.
- Performance metrics used: K-NN (RMSE: 5.91), Decision Trees (RMSE: 1.99), and MLP (RMSE: 0.39).
![Screenshot 2025-02-11 at 12 41 59 p m](https://github.com/user-attachments/assets/728b3af1-01f0-4a44-8244-9adb99211afe)
![Screenshot 2025-02-11 at 12 43 37 p m](https://github.com/user-attachments/assets/a0b3c1c1-ad51-45e9-91df-ad4790dbb6c4)

## Future Improvements

- Enhance feature engineering to improve model accuracy.
- Implement real-time prediction using a web-based API.

## License

This project is open-source and available under the MIT License.

## Acknowledgments

- Data sourced from hospital records and publicly available datasets.
- Inspired by real-world applications in healthcare analytics.


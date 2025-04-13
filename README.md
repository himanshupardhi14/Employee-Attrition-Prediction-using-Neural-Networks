# HR Analytics - Employee Attrition Prediction using Deep Learning

This project is a deep learning-based classification model built using TensorFlow and Keras to predict employee attrition (whether an employee will leave the company) based on various HR-related features.

## 📊 Dataset

The dataset used is `HR_comma_sep.csv`, which contains employee data such as satisfaction level, number of projects, average monthly hours, salary level, etc.

## 🔍 Features Used

The model uses the following features:

- `satisfaction_level`
- `last_evaluation`
- `number_project`
- `average_montly_hours`
- `time_spend_company`
- `Work_accident`
- `promotion_last_5years`
- `sales` (encoded)
- `salary` (encoded)

## 🧪 Target Variable

- `left`: Indicates whether the employee has left the company (1) or not (0).

## ⚙️ Model Architecture

The deep learning model is built using Keras' Sequential API and includes:

- Input layer with 64 neurons and ReLU activation
- Dropout layer (20%) for regularization
- Hidden layer with 32 neurons and ReLU activation
- Dropout layer (20%)
- Output layer with sigmoid activation for binary classification

### Optimizer & Loss

- Optimizer: `Adam`
- Loss Function: `Binary Crossentropy`

## 🚀 Training

The model is trained with:

- 100 epochs
- Batch size: 32
- 70/30 train-test split
- Validation on the test set

## 📈 Results & Visualization

Model training accuracy, validation accuracy, loss, and validation loss are plotted using Matplotlib to help visualize the learning process.

### Example Output:
- Accuracy: ~0.95 (may vary depending on data and parameters)

## 🛠️ Preprocessing

- Label encoding is applied to categorical features (`salary`, `sales`)
- StandardScaler is used to scale numerical features
- Train-test split with `random_state=42` for reproducibility


## 💡 Future Improvements

- Add EarlyStopping and ModelCheckpoint for better training control
- Use GridSearch or RandomSearch for hyperparameter tuning
- Try advanced architectures or ensemble methods

## 📚 Requirements

Install dependencies using:

```bash
pip install pandas scikit-learn tensorflow matplotlib



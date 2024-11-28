# Book Recommendation System

## Overview
This project implements a CNN-based book recommendation system using Keras in Python. It predicts user preferences based on book metadata and ratings.
<p> <img src="https://github.com/user-attachments/assets/d382679b-878c-477b-b95c-c4752653258e" width="1000"> </p>  

## Features
- **CNN Model:** Trained using book rating data.
- **Dataset:** Includes CSV files with book and rating information.
- **Pretrained Models:** Provided for quick evaluation.

## File Structure
- `main.py`: Script to train and evaluate the model.
- `books.csv`, `books2.csv`, `ratings.csv`: Dataset files.
- `regression_model.h5`, `regression_model2.h5`: Pretrained models.

---

## Installation
To run the project, install Python and the required libraries.

### Prerequisites
- Python 3.x
- TensorFlow
- Keras
- NumPy
- Pandas

### Steps
1. **Clone the repository**
   ```bash
   git clone https://github.com/AminLari/Book-Recommendation-System.git
   cd Book-Recommendation-System


2. **Install the dependencies**
   ```bash
   pip install tensorflow keras numpy pandas

## Usage
1. **Prepare the dataset:**
   Ensure the CSV files are in the working directory.

2. **Run the training and evaluation script:**
   To train the CNN model on the dataset, run the following Python script. The script will load the dataset, preprocess the audio data, and train the model.
   ```bash
   python main.py

3. **Customize:**
   Modify main.py to adjust dataset paths and parameters.

## Results
- The system predicts book ratings based on user preferences.
- ou can test its performance using the pretrained models and provided datasets.

## Contact
For questions or suggestions, please contact Amin Lari.

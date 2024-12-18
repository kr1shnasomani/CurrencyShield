<h1 align="center">CurrencyShield</h1>
Designed and implemented a deep learning model using ResNet50 for counterfeit currency detection, featuring advanced image preprocessing, model training and evaluation. Delivered high-accuracy classification with detailed performance metrics and confidence-based visual predictions.

## Execution Guide:
1. Run the following command line in the terminal:
   ```
   pip install numpy matplotlib scikit-learn tensorflow
   ```

2. Download the dataset and copy paste the path of it in the code

3. Upon running the code it saves an additional file named `model.keras` (this file stores the trained model)

4. Copy paste the path of the currency image for which the detection needs to be done

5. The code will display the prediction with its confidence

## Accuracy & Loss Over Epochs:

![image](https://github.com/user-attachments/assets/edf5dfec-3702-4c2a-a20a-444af71ca396)

![image](https://github.com/user-attachments/assets/52b7c934-e46b-485d-8959-046a9ab0e113)

## Model Prediction:

![image](https://github.com/user-attachments/assets/a6170394-8c9b-4906-864e-9c21bf79d885)

![image](https://github.com/user-attachments/assets/24cd2b5c-8212-4da8-a820-8e03fcf9d63b)

## Overview:
The code is designed to build, train, and evaluate a deep learning model for detecting fake currency based on images. Here's an overview of its components and functionality:

1. **Imports**
   - Various libraries are imported, including:
     - `numpy` and `matplotlib` for numerical operations and plotting.
     - `sklearn.metrics` for classification metrics.
     - `tensorflow` and Keras for building and training the deep learning model.
     - `warnings` to suppress warnings.

2. **Data Preprocessing and Augmentation**
   - The `ImageDataGenerator` from Keras is used to apply real-time data augmentation, such as:
     - **Rotation**, **horizontal flip**, and **vertical flip** to help generalize the model.
     - Images are preprocessed using the ResNet50's preprocessing function (`preprocess_input`).
   - Two generators are created: `train_generator` and `validation_generator`, which read images from the respective directories (`train` and `validation`) and apply preprocessing.

3. **Model Definition**
   - The code utilizes **ResNet50** as the base model with pre-trained weights (`imagenet`).
   - The top layers are removed (via `include_top=False`) to allow custom classification layers to be added.
   - The function `build_model` builds a custom model by adding:
     - **Flatten** layer to flatten the output of the base model.
     - Two fully connected (FC) layers with **ReLU activation** and **Dropout** for regularization.
     - A final **softmax** layer for multi-class classification (distinguishing between "Real" and "Fake").
   - The model is compiled with the **SGD optimizer** and **categorical cross-entropy loss**.

4. **Model Training**
   - The model is trained for **10 epochs** with the training and validation data generators.
   - **ModelCheckpoint** saves the best model based on validation accuracy.
   - **EarlyStopping** monitors the validation accuracy and stops training if it doesn’t improve for 40 consecutive epochs to avoid overfitting.
   
5. **Model Evaluation**
   - After training, the `create_classification_report` function is called to evaluate the model's performance on the validation dataset:
     - **Accuracy score** and **classification report** are generated, showing detailed metrics such as precision, recall, and F1-score.

6. **Training and Validation Loss/Accuracy Visualization**
   - The **training and validation accuracy** and **loss** over the epochs are plotted using `matplotlib`.

7. **Prediction on New Images**
   - The `predict_currency` function predicts whether a currency note (image) is "Real" or "Fake":
     - The image is loaded, resized to 300x300 pixels, and preprocessed.
     - The model outputs a prediction, and based on the confidence score:
       - If confidence is less than 0.5, it predicts "Real".
       - If confidence is greater than 0.5, it predicts "Fake".
     - The result is displayed as an image with the predicted label and confidence score.

8. **Example Prediction**
   - An example image (`Fake.jpeg`) is tested using the `predict_currency` function, and the result is displayed with the prediction and confidence score.

### Key Features:
- **Data Augmentation** to improve generalization.
- **Transfer Learning** using ResNet50.
- **EarlyStopping** and **ModelCheckpoint** for optimal training.
- **Classification Report** to evaluate model performance.
- **Real-time Prediction** for classifying images as "Real" or "Fake".

This code is primarily aimed at detecting fake currency using image classification, leveraging deep learning techniques like transfer learning with ResNet50.

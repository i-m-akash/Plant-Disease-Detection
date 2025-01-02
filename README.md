# Plant Disease Detection Web App

## Description
The **Plant Disease Detection Web App** is a Streamlit-based application that allows users to detect plant diseases by uploading images of plant leaves. It provides descriptions and possible treatment links for the detected disease in the selected plant type. The application also supports multiple languages for user convenience.

---

## Features
- **Multi-language Support**: Uses Google Translate API to translate UI and results into different languages.
- **Plant Disease Detection**: Identifies diseases in plants such as Potato, Grape, Apple, and Pepper Bell.
- **Detailed Information**: Provides descriptions and treatment options for detected diseases.
- **Image Upload**: Allows users to upload leaf images in `.jpg`, `.jpeg`, and `.png` formats.
- **Confidence Score**: Displays the model’s confidence level for the prediction.

---

## Tech Stack
- **Frontend**: [Streamlit](https://streamlit.io/)
- **Machine Learning**: TensorFlow for image classification.
- **Translation**: Google Translate API
- **Backend**: Python

---

## Installation
### Prerequisites
1. Python 3.7+
2. Required Python Libraries:
   - streamlit
   - tensorflow
   - pillow
   - googletrans==4.0.0-rc1

### Steps
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. Open the app in your browser at `http://localhost:8501`.

---

## How to Use
1. Launch the app using the installation steps.
2. Select your preferred language from the dropdown menu.
3. Choose the plant type (e.g., Potato, Grape, Pepper Bell, Apple).
4. Upload a leaf image using the file uploader.
5. View the predicted disease, confidence level, description, and treatment options.

---

## Supported Plants and Diseases
1. **Potato**
   - Early Blight
   - Late Blight
   - Healthy

2. **Grape**
   - Black Rot
   - Esca (Black Measles)
   - Grape Leaf Blight (Isariopsis Leaf Spot)
   - Healthy

3. **Pepper Bell**
   - Bacterial Spot
   - Healthy

4. **Apple**
   - Apple Scab
   - Black Rot
   - Cedar Apple Rust
   - Healthy

---

## Results
- Achieved an accuracy of 96% on the validation dataset.
- Confusion matrix and class-wise accuracy available in the evaluation report.

---

## Future Enhancements
- **Expand Dataset**: Add more classes and samples for better generalization.
- **Real-time Integration**: Integrate with IoT devices for field applications.
- **Mobile App Deployment**: Use TensorFlow Lite for Android/iOS applications.

---

## Dataset
The dataset used is the PlantVillage Dataset or any dataset containing labeled images of multiple plant diseases (e.g., potato, grape, apple, pepper bell).
Ensure the dataset is organized into train, validation, and test directories with subfolders for each class (e.g., healthy, early_blight, late_blight for potato; black_rot, esca for grape).

---

## Contributing
Feel free to contribute to this project by opening issues or submitting pull requests.

---

## References
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Google Translate API](https://cloud.google.com/translate/docs)


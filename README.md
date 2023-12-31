# Deep Learning - Celebrity Look-A-like Project

## Project Description

This project aims to build a Celebrity Look-Alike Model using CNN (Convolutional Neural Networks) based on the VGG (Visual Geometry Group) 16 Model. Users can enjoy a lighthearted exploration of their celebrity doppelgängers. The project integrates image processing, CNN implementation, transfer learning and practical neural network applications, aligning with the course's focus on the practical implementation of Deep Neural Networks.

## Methodology (Tech Stack)

-   **Python** 3.9.18 (64-bit)
    -   IPython 8.18.1
    -   Jupyter Notebook 6.4.4
-   **NumPy** 1.19.5 used for dealing with arrays and matrices of numbers.
-   **Pandas** 1.3.4 used for data manipulation and analysis.
-   **Matplotlib** 3.6.0 used for data visualization.
-   **Scipy** 1.10.0 used for loading MATLAB files.
-   **TensorFlow-gpu** 2.6.0 used for building and training the model.
-   **Keras** 2.6.0 used for integrating TensorFlow and building the model.
-   **CV2** 4.5.3 used for image loading, preprocessing, face detection, and visualization.
-   **Chime** 0.7.0 used for audio notifications.

All the above packages can be installed using the following command:

```bash
pip install -r requirements.txt
```

## Data-Set

The dataset used in this project is provided by ETH Zurich University. Originally shared by researchers for age and gender prediction tasks, the IMDB Faces dataset will be exclusively utilized. The dataset is structured in the MATLAB file format (.mat), and the celebrity images are stored in the JPG format.

The dataset can be downloaded from the following link: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar

The dataset comprises approximately 460,000 records, with an overall size of approximately 6.5 GB.

Data-set Example:

![IMDB Dataset](./Images/Imdb%20crop%20teaser.png)

## Project Structure

The project is structured as follows:

-   **Including Necessary Libraries:** Importing the necessary libraries found in the requirements.txt file.

-   **Loading the Dataset:** Loading the training dataset from the .mat file and converting it to a Pandas DataFrame, and then loading the test dataset from the test_images folder, and converting it to a Pandas DataFrame.

-   **Data Cleaning:** Removing unnecessary columns from the training and test datasets, and then removing the records with missing and duplicate values.

-   **Data Visualization:** Visualizing the training dataset using Matplotlib.

-   **Data Preprocessing:** Preprocessing the training and test datasets by resizing the images, extracting the faces using the Haar Cascade Classifier, and then converting the images to a NumPy array.

    -   **Haar Cascade Classifier:** The Haar Cascade Classifier is a machine learning-based approach where a cascade function is trained from a lot of positive and negative images. It is then used to detect objects and faces in other images.

    -   **Preprocessing Example:**

        ![Preprocessing Example](./Images/Normalized%20Images.png)

-   **Building the Model:** Building the model using the VGG16 model structure, and excluding the last 3 layers (the fully connected layers), to include only the output of the flatten layer which is the feature vector extracted from the images.

    -   **VGG16 Face Descriptor Model Structure:**

        ![VGG16 Model Structure](./Images/Vgg%20face%20descriptor%20Model%20Plot.png)

    -   **Adding the Model Weights:** Adding the weights of the VGG16 model to the model, the weights can be downloaded from the following link: https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5

-   **Predicting the Faces' Features:** Predicting the features of the faces in the training and test datasets using the model.

-   **Finding the Best Match:** Finding the best match for each face in the test dataset by calculating the Cosine Similarity between the features of the faces in the training and test datasets.

    -   **Cosine Similarity:** Cosine Similarity is a measure of similarity between two non-zero vectors of an inner product space. It is defined to equal the cosine of the angle between them, if the cosine is zero, the vectors are orthogonal to each other, and if the cosine is one, the vectors are completely similar, and if the cosine is negative one, the vectors are completely dissimilar.

    -   **Cosine Similarity Formula:**

        ![Cosine Similarity Formula](./Images/Cosine%20similarity.png)

    -   **Finding the Best Match Example for the Test Images:** !

        ![Best Match Example](./Images/Test%20img%20vs%20Celeb%20img%20output.png)

    -   **Finding the Best Match Example for the Celebrities Images:**

        ![Best Match Example](./Images/Celeb%20img%20vs%20Celeb%20img%20output.png)

-   **Getting the Best Match in Real-Time:** Getting the best match for the face in real-time using the webcam.

    -   **Real-Time Best Match Example:** !

        ![Real-Time Best Match Example](./Images/Real%20time%20output.png)

## References

-   **VGG16 Model Architecture:**
    https://github.com/fchollet/deep-learning-models/blob/master/vgg16.py

-   **IMDB-WIKI Dataset:**
    https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/

-   **Haar Cascade Classifier:**
    https://docs.opencv.org/3.4/db/d28tutorial_cascade_classifier.html

-   **Cosine Similarity:**
    https://en.wikipedia.org/wiki/Cosine_similarity

-   **Project Inspiration:**
    https://www.youtube.com/watch?v=jaxkEn-Kieo

-   **Real-Time Best Match:**
    https://www.youtube.com/watch?v=RMgIKU1H8DY

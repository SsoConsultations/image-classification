import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import os
import shutil
import io # For handling in-memory files

# Set page config for Streamlit app
st.set_page_config(
    page_title="Custom Image Classifier",
    page_icon="🖼️",
    layout="wide"
)

# --- Constants ---
IMAGE_SIZE = (224, 224)
MODEL_SAVE_DIR = 'model_artifacts' # Directory to save model and class names
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'my_custom_classifier_model.keras')
CLASS_NAMES_PATH = os.path.join(MODEL_SAVE_DIR, 'class_names.txt')

# Ensure model save directory exists
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


# --- Helper Functions (to encapsulate logic) ---

@st.cache_resource # Cache the model loading for efficiency
def load_model_and_class_names():
    """Loads the trained model and class names, if they exist."""
    model = None
    class_names = []
    if os.path.exists(MODEL_PATH) and os.path.exists(CLASS_NAMES_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            with open(CLASS_NAMES_PATH, 'r') as f:
                class_names = [line.strip() for line in f]
            st.success("Pre-trained model and class names loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model or class names: {e}")
            st.exception(e) # Display the full exception for debugging
    return model, class_names

def preprocess_image_for_prediction(image_bytes):
    """Loads, resizes, and pre-processes an image for MobileNetV2 prediction."""
    img = Image.open(io.BytesIO(image_bytes)).resize(IMAGE_SIZE)
    img_array = np.array(img)
    if img_array.ndim == 2:  # Grayscale
        img_array = np.stack((img_array,) * 3, axis=-1)
    elif img_array.shape[-1] == 4:  # RGBA
        img_array = img_array[..., :3]
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    return tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

# --- Streamlit UI ---

st.title("Custom Image Classifier with Transfer Learning")
st.markdown("Build and use your own image classification model right here!")

# Initialize session state variables
if 'model' not in st.session_state:
    st.session_state.model, st.session_state.class_names = load_model_and_class_names()

if 'training_data_collected' not in st.session_state:
    st.session_state.training_data_collected = {} # Stores {'label': [list_of_image_bytes]}

if 'current_class_num' not in st.session_state:
    st.session_state.current_class_num = 1 # Tracks which class the user is currently uploading

# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Train New Model", "Predict with Model"])

if page == "Train New Model":
    st.header("📚 Train a New Custom Model")
    st.info("Upload images for each class. You need at least two classes.")
    st.warning("Training can take some time depending on your dataset size and hardware.")

    if st.button("Clear All Uploaded Training Data", key="clear_training_data"):
        st.session_state.training_data_collected = {}
        st.session_state.current_class_num = 1
        st.success("All uploaded training data cleared. Please start uploading your first class.")
        st.experimental_rerun() # Rerun to clear the display

    # Dynamic Class Data & Labels - Modified for guided flow
    st.subheader("1. Define Classes & Upload Images")

    # Display current collected classes and allow new ones
    if st.session_state.training_data_collected:
        st.markdown("**Currently defined classes:**")
        for label, images in st.session_state.training_data_collected.items():
            st.write(f"- **{label}**: {len(images)} images")
        st.markdown("---")

    # Prompt for current class upload
    current_class_prompt = f"Please enter the label for Class {st.session_state.current_class_num} (e.g., 'cats', 'dogs', 'other'):"
    new_class_label = st.text_input(current_class_prompt, key=f"label_input_{st.session_state.current_class_num}").strip()

    if new_class_label:
        if new_class_label in st.session_state.training_data_collected:
            st.warning(f"Label '{new_class_label}' already used. Please enter a unique label for Class {st.session_state.current_class_num}.")
        else:
            st.write(f"Now, please upload all images for '{new_class_label}'.")
            st.info("You can select multiple files at once in the file dialog.")
            uploaded_files = st.file_uploader(
                f"Upload images for '{new_class_label}'",
                type=["png", "jpg", "jpeg", "gif", "bmp"],
                accept_multiple_files=True,
                key=f"uploader_{st.session_state.current_class_num}" # Unique key for each uploader
            )

            if uploaded_files:
                st.session_state.training_data_collected[new_class_label] = []
                for uploaded_file in uploaded_files:
                    image_bytes = uploaded_file.read()
                    st.session_state.training_data_collected[new_class_label].append(image_bytes)
                st.success(f"Successfully uploaded {len(uploaded_files)} images for '{new_class_label}'.")

                # Increment class number for the next upload
                st.session_state.current_class_num += 1

                # Clear the input box and uploader for the next class
                if len(st.session_state.training_data_collected) >= 2:
                    st.info(f"You have defined {len(st.session_state.training_data_collected)} classes. If you have another dataset, please define a new class above. Otherwise, you can click 'Start Training' below.")
                else:
                     st.info(f"Please upload at least one more dataset to have a minimum of two classes for training.")
                st.experimental_rerun() # Rerun to show updated counts and new input prompt
    elif st.session_state.current_class_num > 1: # Only if at least one class has been processed
         if len(st.session_state.training_data_collected) >= 2:
             st.info(f"You have defined {len(st.session_state.training_data_collected)} classes. If you have another dataset, please define a new class above. Otherwise, you can click 'Start Training' below.")
         else:
              st.info(f"You have defined {len(st.session_state.training_data_collected)} class. Please upload at least one more dataset to have a minimum of two classes for training.")


    if len(st.session_state.training_data_collected) < 2:
        st.warning("You need to define and upload images for at least two classes to proceed with training.")

    # Training Button
    if st.button("Start Training", disabled=(len(st.session_state.training_data_collected) < 2), key="start_training_btn"):
        st.subheader("2. Preparing Data and Training Model")
        with st.spinner("Preparing data and training model... This may take a while."):
            all_images = []
            all_labels = []

            # Progress bar for image loading
            total_images_to_load = sum(len(v) for v in st.session_state.training_data_collected.values())
            progress_text = "Image processing in progress. Please wait."
            image_load_bar = st.progress(0, text=progress_text)
            images_loaded_count = 0

            # Flatten collected data
            for label, image_bytes_list in st.session_state.training_data_collected.items():
                for image_bytes in image_bytes_list:
                    try:
                        img = Image.open(io.BytesIO(image_bytes)).resize(IMAGE_SIZE)
                        img_array = np.array(img)
                        if img_array.ndim == 2:  # Grayscale
                            img_array = np.stack((img_array,) * 3, axis=-1)
                        elif img_array.shape[-1] == 4:  # RGBA
                            img_array = img_array[..., :3]
                        all_images.append(img_array)
                        all_labels.append(label)
                    except Exception as e:
                        st.warning(f"Could not load an image for label '{label}': {e}")
                        # Display specific error for the image
                        st.exception(e)
                        continue
                    images_loaded_count += 1
                    image_load_bar.progress(images_loaded_count / total_images_to_load, text=f"{progress_text} ({images_loaded_count}/{total_images_to_load})")
            image_load_bar.empty() # Clear the progress bar after completion

            if not all_images:
                st.error("No valid images were loaded for training. Please check your uploads.")
                st.stop()

            all_images = np.array(all_images)

            le = LabelEncoder()
            numerical_labels = le.fit_transform(all_labels)
            st.session_state.class_names = le.classes_.tolist() # Store as list in session state
            st.write(f"Labels encoded: {list(st.session_state.class_names)}")

            X_train, X_val, y_train, y_val = train_test_split(
                all_images, numerical_labels, test_size=0.2, random_state=42, stratify=numerical_labels
            )

            st.write(f"Total images loaded: {len(all_images)}")
            st.write(f"Training images: {len(X_train)}")
            st.write(f"Validation images: {len(X_val)}")

            # --- MODIFIED: Simplified ImageDataGenerator parameters ---
            train_datagen = ImageDataGenerator(
                preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
                horizontal_flip=True, # Keep common, safe augmentation
                # Removed or simplified other aggressive augmentations for troubleshooting
                # rotation_range=0,
                # width_shift_range=0,
                # height_shift_range=0,
                # shear_range=0,
                # zoom_range=0,
                # vertical_flip=False,
                # brightness_range=None,
                # fill_mode='nearest' # Only relevant if shifts are used
            )
            val_datagen = ImageDataGenerator(
                preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
            )
            # --- END MODIFIED ---

            train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
            validation_generator = val_datagen.flow(X_val, y_val, batch_size=32)

            # Model Building (Transfer Learning)
            base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
            for layer in base_model.layers[:-50]:
                layer.trainable = False
            for layer in base_model.layers[-50:]:
                layer.trainable = True

            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(128, activation='relu')(x)
            x = Dropout(0.5)(x)
            predictions = Dense(len(st.session_state.class_names), activation='softmax')(x)
            model = Model(inputs=base_model.input, outputs=predictions)
            model.compile(optimizer=Adam(learning_rate=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            st.subheader("Model Summary")
            # You can capture and display the model summary like this if desired:
            # import io
            # from contextlib import redirect_stdout
            # with io.StringIO() as s, redirect_stdout(s):
            #     model.summary()
            # st.text(s.getvalue())

            # Model Training
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            # --- CHANGE APPLIED HERE: Using st.spinner and verbose=1 ---
            with st.spinner("Training the model... This might take some time, please check the Streamlit logs for live epoch updates."):
                history = model.fit(
                    train_generator,
                    epochs=100,
                    validation_data=validation_generator,
                    callbacks=[early_stopping],
                    verbose=1 # Changed to verbose=1 to allow Keras to print to stdout/logs
                )
            # --- END OF CHANGE ---

            st.session_state.model = model # Store the trained model in session state

            # Evaluation
            loss, accuracy = model.evaluate(validation_generator, verbose=0)
            st.subheader("3. Model Evaluation")
            st.success(f"Validation Loss: {loss:.4f}")
            st.success(f"Validation Accuracy: {accuracy*100:.2f}%")

            # Save the Model and Class Names
            st.subheader("4. Saving Model")
            try:
                model.save(MODEL_PATH)
                with open(CLASS_NAMES_PATH, 'w') as f:
                    for name in st.session_state.class_names:
                        f.write(f"{name}\n")
                st.success(f"Model saved to '{MODEL_PATH}'")
                st.success(f"Class names saved to '{CLASS_NAMES_PATH}'")
            except Exception as e:
                st.error(f"Error saving model or class names: {e}")
                st.exception(e)

            st.balloons() # Visual celebration
            st.success("🥳 Training completed successfully! Your custom model is ready.")
            st.info("You have successfully completed training. You can now switch to the 'Predict with Model' section in the sidebar to test it with new images.")

elif page == "Predict with Model":
    st.header("🔮 Predict Image Class")

    if st.session_state.model is None or not st.session_state.class_names:
        st.warning("No model found. Please train a model first using the 'Train New Model' section.")
    else:
        st.write("Upload an image to get a prediction from your trained model.")
        uploaded_predict_file = st.file_uploader(
            "Upload an image for classification",
            type=["png", "jpg", "jpeg", "gif", "bmp"],
            key="predict_file_uploader"
        )

        if uploaded_predict_file is not None:
            st.image(uploaded_predict_file, caption="Uploaded Image", use_column_width=True)

            with st.spinner("Classifying image..."):
                try:
                    image_bytes = uploaded_predict_file.read()
                    preprocessed_img = preprocess_image_for_prediction(image_bytes)

                    predictions = st.session_state.model.predict(preprocessed_img)
                    predicted_class_index = np.argmax(predictions, axis=1)[0]
                    predicted_probability = np.max(predictions, axis=1)[0]
                    sorted_probs = np.sort(predictions[0])[::-1]

                    st.subheader("Prediction Results:")

                    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.9, 0.05, key="confidence_slider") # Slider for threshold

                    # Check for uncertainty or low confidence
                    if len(st.session_state.class_names) > 1 and (sorted_probs[0] - sorted_probs[1]) < 0.2:
                        st.warning("The model is somewhat uncertain about this image (top two probabilities are close).")
                        st.write(f"**Highest predicted class:** '{st.session_state.class_names[predicted_class_index]}' with **{predicted_probability*100:.2f}%** confidence.")
                        if len(st.session_state.class_names) > 1:
                            st.write(f"Second highest: '{st.session_state.class_names[np.argsort(predictions[0])[-2]]}' with {sorted_probs[1]*100:.2f}% confidence.")
                        st.info("This image might not clearly belong to any trained class, or it could be an 'other' type.")
                    elif predicted_probability >= confidence_threshold:
                        predicted_label = st.session_state.class_names[predicted_class_index]
                        st.success(f"The model confidently predicts this image is a **'{predicted_label}'** with **{predicted_probability*100:.2f}%** confidence.")
                    else:
                        st.info("The model cannot confidently classify this image.")
                        st.write(f"Highest predicted class: '{st.session_state.class_names[predicted_class_index]}' with {predicted_probability*100:.2f}% confidence (below {confidence_threshold*100:.0f}% threshold).")
                        st.write("This image may not belong to any trained class, or the model needs more diverse training data.")

                    st.markdown("---")
                    st.write("All Class Probabilities:")
                    for i, prob in enumerate(predictions[0]):
                        st.write(f"- {st.session_state.class_names[i]}: {prob*100:.2f}%")

                except Exception as e:
                    st.error(f"Error classifying image: {e}")
                    st.error("Please ensure the uploaded file is a valid image.")
                    st.exception(e) # Display the full exception for debugging

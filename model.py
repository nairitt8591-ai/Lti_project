from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, GRU, Dense, Dropout, TimeDistributed,
    RandomFlip, RandomRotation, RandomZoom, RandomContrast
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2

def build_contextual_model(sequence_length, frame_shape, num_classes):
    """
    Builds a Convolutional Recurrent Neural Network (CRNN) to understand
    sequences of hand signs, with augmentations and dropout to reduce overfitting.
    """
    video_input = Input(shape=(sequence_length, *frame_shape), name="video_input")

    # --- Data Augmentation Layers ---
    # These layers create variations of the training images to make the model more robust.
    x = TimeDistributed(RandomFlip('horizontal'))(video_input)
    x = TimeDistributed(RandomRotation(0.2))(x)
    x = TimeDistributed(RandomZoom(0.2))(x)
    x = TimeDistributed(RandomContrast(0.2))(x)

    # --- CNN Feature Extractor (The "Eyes") ---
    # We use a pre-trained MobileNetV2 for powerful feature extraction.
    cnn_base = MobileNetV2(
        input_shape=frame_shape,
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    # Freeze the pre-trained layers to prevent them from being changed during initial training.
    cnn_base.trainable = False

    # Apply the CNN to each frame of the sequence.
    encoded_frames = TimeDistributed(cnn_base, name="frame_feature_extractor")(x)

    # --- RNN Sequence Processor (The "Brain") ---
    # A GRU processes the sequence of features from the CNN to understand context.
    encoded_sequence = GRU(256, return_sequences=False, name="sequence_processor")(encoded_frames)
    
    # --- Classifier Head with Dropout ---
    # Dropout layers help prevent the model from memorizing the training data (overfitting).
    dropout_layer_1 = Dropout(0.5)(encoded_sequence)
    dense_layer = Dense(128, activation='relu')(dropout_layer_1)
    dropout_layer_2 = Dropout(0.5)(dense_layer)
    output_layer = Dense(num_classes, activation='softmax', name="classifier_output")(dropout_layer_2)

    # --- Construct and Compile the Model ---
    model = Model(inputs=video_input, outputs=output_layer, name="Contextual_CRNN_Model")

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
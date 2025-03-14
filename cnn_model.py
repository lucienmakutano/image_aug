from tensorflow.keras import layers, models

# This function creates our model that learns to tell tire pictures apart
def create_cnn(input_shape, num_classes):
    # Start making the model
   _model = models.Sequential([
        # First layer: checks picture features like edges and patterns
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),  # 32 small filters
        layers.MaxPooling2D((2, 2)),  # Makes the picture smaller but keeps important details

        # Second layer: looks deeper into the picture features
        layers.Conv2D(64, (3, 3), activation='relu'),  # 64 filters for clearer details
        layers.MaxPooling2D((2, 2)),  # Makes it smaller again

        # Third layer: digs deeper to get even finer details
        layers.Conv2D(128, (3, 3), activation='relu'),  # Adds more filters for complex stuff
        layers.MaxPooling2D((2, 2)),  # Shrinks it down one more time

        # Flattens all the info so it can pass to the next part
        layers.Flatten(),

        # Fully connected layer that brings all the info together
        layers.Dense(128, activation='relu'),

        # Adds some randomness to stop it from cramming too much info
        layers.Dropout(0.5),

        # Final layer gives the chances of each class like cracked or normal tire
        layers.Dense(num_classes, activation='softmax')
    ])

   _model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

   return _model

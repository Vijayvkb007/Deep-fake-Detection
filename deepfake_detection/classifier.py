import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from config import *

class DeepfakeClassifier:
    def __init__(self, input_dim=FEATURE_DIM):
        self.model = self._build_classifier(input_dim)
        
    def _build_classifier(self, input_dim):
        """Build binary classifier model"""
        model = Sequential([
            Dense(256, activation='relu', input_dim=input_dim,
                 kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train the classifier"""
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(
                f"{MODEL_SAVE_PATH}/classifier.h5",
                save_best_only=True,
                monitor='val_accuracy'
            ),
            tf.keras.callbacks.TensorBoard(log_dir='logs')
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks
        )
        
        return history
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        return self.model.evaluate(X_test, y_test)
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import json

class SustainabilityClassifier:
    def __init__(self):
        # Define sustainability features to detect
        self.sustainability_features = {
            'recyclable_materials': ['plastic', 'paper', 'glass', 'metal'],
            'eco_packaging': ['minimal_packaging', 'biodegradable', 'recycled_content'],
            'certifications': ['energy_star', 'forest_stewardship', 'fairtrade', 'organic']
        }
        
        # Initialize the model
        self.model = self._build_model()
        
    def _build_model(self):
        # Base model using EfficientNetB0 (pre-trained on ImageNet)
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=(224, 224, 3)
        )
        
        # Freeze the base model
        base_model.trainable = False
        
        # Build the complete model
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dense(len(self.sustainability_features) * 2, activation='sigmoid')
        ])
        
        return model
    
    def preprocess_image(self, image_path):
        """Preprocess image for model input"""
        # Load and resize image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        
        # Normalize pixel values
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        return np.expand_dims(img, axis=0)
    
    def analyze_sustainability(self, image_path):
        """Analyze product sustainability from image"""
        # Preprocess image
        processed_image = self.preprocess_image(image_path)
        
        # Get model predictions
        predictions = self.model.predict(processed_image)
        
        # Process and structure the results
        sustainability_score = self._calculate_sustainability_score(predictions[0])
        feature_analysis = self._analyze_features(predictions[0])
        
        return {
            'overall_sustainability_score': sustainability_score,
            'feature_analysis': feature_analysis,
            'recommendations': self._generate_recommendations(feature_analysis)
        }
    
    def _calculate_sustainability_score(self, predictions):
        """Calculate overall sustainability score"""
        return float(np.mean(predictions) * 100)
    
    def _analyze_features(self, predictions):
        """Analyze individual sustainability features"""
        feature_analysis = {}
        idx = 0
        
        for category, features in self.sustainability_features.items():
            category_scores = {}
            for feature in features:
                score = float(predictions[idx] * 100)
                confidence = float(predictions[idx + 1] * 100)
                category_scores[feature] = {
                    'score': score,
                    'confidence': confidence
                }
                idx += 2
            feature_analysis[category] = category_scores
            
        return feature_analysis
    
    def _generate_recommendations(self, feature_analysis):
        """Generate sustainability improvement recommendations"""
        recommendations = []
        
        # Analyze each category and generate specific recommendations
        for category, features in feature_analysis.items():
            for feature, scores in features.items():
                if scores['score'] < 50 and scores['confidence'] > 70:
                    recommendations.append(
                        f"Consider improving {feature.replace('_', ' ')} "
                        f"in the {category.replace('_', ' ')} category"
                    )
        
        return recommendations

    def train(self, training_data, labels, epochs=10):
        """Train the model with custom dataset"""
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model.fit(
            training_data,
            labels,
            epochs=epochs,
            validation_split=0.2
        )
    
    def save_model(self, file_path):
        """Save the trained model to a file"""
        self.model.save(file_path)
        print(f"Model saved to {file_path}")
    
    def load_model(self, file_path):
        """Load a trained model from a file"""
        self.model = tf.keras.models.load_model(file_path)
        print(f"Model loaded from {file_path}")


if __name__ == "__main__":
    # Example workflow
    classifier = SustainabilityClassifier()

    # Train or load model
    model_path = "trained_sustainability_model.h5"
    if tf.io.gfile.exists(model_path):
        classifier.load_model(model_path)
    else:
        # Example: Generate random dataset for training
        training_data = np.random.rand(100, 224, 224, 3).astype(np.float32)
        labels = np.random.rand(100, len(classifier.sustainability_features) * 2).astype(np.float32)

        print("Training the model...")
        classifier.train(training_data, labels, epochs=5)
        classifier.save_model(model_path)

    # Analyze an image
    image_path = "path/to/your/product_image.jpg"  # Replace with an actual image path
    results = classifier.analyze_sustainability(image_path)

    # Print results
    print(json.dumps(results, indent=4))

    # Upload product image
    uploaded = files.upload()

    # Get the image file path
    image_path = list(uploaded.keys())[0]
    print(f"Uploaded image: {image_path}")

    results = classifier.analyze_sustainability(image_path)

    # Print results
    print(json.dumps(results, indent=4))

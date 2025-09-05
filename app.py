import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os
from model_utils import create_cnn_model, train_model
from preprocessing import preprocess_data, decode_review
from visualization import plot_training_history, plot_confusion_matrix, plot_word_distribution

# Set page configuration
st.set_page_config(
    page_title="CNN Sentiment Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'word_index' not in st.session_state:
    st.session_state.word_index = None
if 'history' not in st.session_state:
    st.session_state.history = None
if 'x_test' not in st.session_state:
    st.session_state.x_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None

def main():
    st.title("🎬 CNN-Based Sentiment Analysis")
    st.markdown("### IMDB Movie Reviews Classification")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Data Overview", "Model Training", "Model Evaluation", "Live Prediction", "Model Architecture"]
    )
    
    if page == "Data Overview":
        data_overview_page()
    elif page == "Model Training":
        model_training_page()
    elif page == "Model Evaluation":
        model_evaluation_page()
    elif page == "Live Prediction":
        live_prediction_page()
    elif page == "Model Architecture":
        model_architecture_page()

def data_overview_page():
    st.header("📊 Data Overview")
    
    with st.spinner("Loading IMDB dataset..."):
        try:
            # Load IMDB dataset
            max_features = st.sidebar.slider("Max Features (Vocabulary Size)", 1000, 20000, 10000)
            maxlen = st.sidebar.slider("Max Sequence Length", 50, 1000, 500)
            
            (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
            word_index = imdb.get_word_index()
            
            # Store in session state
            st.session_state.word_index = word_index
            st.session_state.x_test = x_test
            st.session_state.y_test = y_test
            
            # Display dataset statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Training Samples", len(x_train))
            with col2:
                st.metric("Test Samples", len(x_test))
            with col3:
                st.metric("Vocabulary Size", max_features)
            with col4:
                st.metric("Max Sequence Length", maxlen)
            
            # Class distribution
            st.subheader("Class Distribution")
            train_positive = np.sum(y_train)
            train_negative = len(y_train) - train_positive
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(
                    x=['Negative', 'Positive'],
                    y=[train_negative, train_positive],
                    title="Training Set Distribution",
                    color=['Negative', 'Positive'],
                    color_discrete_map={'Negative': '#ff6b6b', 'Positive': '#4ecdc4'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                test_positive = np.sum(y_test)
                test_negative = len(y_test) - test_positive
                fig = px.bar(
                    x=['Negative', 'Positive'],
                    y=[test_negative, test_positive],
                    title="Test Set Distribution",
                    color=['Negative', 'Positive'],
                    color_discrete_map={'Negative': '#ff6b6b', 'Positive': '#4ecdc4'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Sample reviews
            st.subheader("Sample Reviews")
            review_idx = st.slider("Select Review Index", 0, min(100, len(x_train)-1), 0)
            
            decoded_review = decode_review(x_train[review_idx], word_index)
            sentiment = "Positive 😊" if y_train[review_idx] == 1 else "Negative 😔"
            
            st.write(f"**Sentiment:** {sentiment}")
            st.write(f"**Review:** {decoded_review}")
            
            # Sequence length distribution
            st.subheader("Sequence Length Distribution")
            lengths = [len(x) for x in x_train[:1000]]  # Sample first 1000 for performance
            fig = px.histogram(
                x=lengths,
                nbins=50,
                title="Distribution of Review Lengths (Sample of 1000)",
                labels={'x': 'Sequence Length', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")

def model_training_page():
    st.header("🚀 Model Training")
    
    if st.session_state.word_index is None:
        st.warning("Please visit the 'Data Overview' page first to load the dataset.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Training Parameters")
        max_features = st.slider("Max Features", 1000, 20000, 10000, key="train_max_features")
        maxlen = st.slider("Max Sequence Length", 50, 1000, 500, key="train_maxlen")
        embedding_dim = st.slider("Embedding Dimension", 32, 256, 128)
        filters = st.slider("Number of Filters", 32, 256, 64)
        kernel_size = st.slider("Kernel Size", 3, 7, 3)
        dropout_rate = st.slider("Dropout Rate", 0.1, 0.8, 0.5)
    
    with col2:
        st.subheader("Training Configuration")
        epochs = st.slider("Epochs", 1, 20, 5)
        batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=2)
        validation_split = st.slider("Validation Split", 0.1, 0.3, 0.2)
        learning_rate = st.selectbox("Learning Rate", [0.001, 0.01, 0.1], index=0)
    
    if st.button("Start Training", type="primary"):
        with st.spinner("Loading and preprocessing data..."):
            try:
                # Load and preprocess data
                (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
                x_train = pad_sequences(x_train, maxlen=maxlen)
                x_test = pad_sequences(x_test, maxlen=maxlen)
                
                # Store test data in session state
                st.session_state.x_test = x_test
                st.session_state.y_test = y_test
                
                # Create model
                model = create_cnn_model(
                    max_features=max_features,
                    maxlen=maxlen,
                    embedding_dim=embedding_dim,
                    filters=filters,
                    kernel_size=kernel_size,
                    dropout_rate=dropout_rate,
                    learning_rate=learning_rate
                )
                
                st.success("Model created successfully!")
                
                # Training progress containers
                progress_bar = st.progress(0)
                status_text = st.empty()
                metrics_container = st.empty()
                
                # Train model
                history = train_model(
                    model, x_train, y_train, 
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=validation_split,
                    progress_callback=lambda epoch, logs: update_training_progress(
                        epoch, epochs, logs, progress_bar, status_text, metrics_container
                    )
                )
                
                # Store model and history in session state
                st.session_state.model = model
                st.session_state.history = history
                
                st.success("Training completed!")
                
                # Display final metrics
                final_accuracy = history.history['accuracy'][-1]
                final_val_accuracy = history.history['val_accuracy'][-1]
                final_loss = history.history['loss'][-1]
                final_val_loss = history.history['val_loss'][-1]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Final Training Accuracy", f"{final_accuracy:.4f}")
                with col2:
                    st.metric("Final Validation Accuracy", f"{final_val_accuracy:.4f}")
                with col3:
                    st.metric("Final Training Loss", f"{final_loss:.4f}")
                with col4:
                    st.metric("Final Validation Loss", f"{final_val_loss:.4f}")
                
                # Plot training history
                st.subheader("Training History")
                fig = plot_training_history(history)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error during training: {str(e)}")

def update_training_progress(epoch, total_epochs, logs, progress_bar, status_text, metrics_container):
    progress = (epoch + 1) / total_epochs
    progress_bar.progress(progress)
    status_text.text(f"Epoch {epoch + 1}/{total_epochs}")
    
    if logs:
        with metrics_container.container():
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Training Accuracy", f"{logs.get('accuracy', 0):.4f}")
            with col2:
                st.metric("Validation Accuracy", f"{logs.get('val_accuracy', 0):.4f}")
            with col3:
                st.metric("Training Loss", f"{logs.get('loss', 0):.4f}")
            with col4:
                st.metric("Validation Loss", f"{logs.get('val_loss', 0):.4f}")

def model_evaluation_page():
    st.header("📈 Model Evaluation")
    
    if st.session_state.model is None:
        st.warning("Please train a model first in the 'Model Training' page.")
        return
    
    model = st.session_state.model
    x_test = st.session_state.x_test
    y_test = st.session_state.y_test
    
    if x_test is None or y_test is None:
        st.warning("Test data not available. Please reload the dataset in 'Data Overview' page.")
        return
    
    with st.spinner("Evaluating model on test set..."):
        try:
            # Make predictions
            predictions = model.predict(x_test)
            y_pred = (predictions > 0.5).astype(int).flatten()
            
            # Calculate metrics
            test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
            
            # Display overall metrics
            st.subheader("Overall Performance")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Test Accuracy", f"{test_accuracy:.4f}")
            with col2:
                st.metric("Test Loss", f"{test_loss:.4f}")
            with col3:
                st.metric("Total Test Samples", len(y_test))
            
            # Classification Report
            st.subheader("Detailed Classification Report")
            report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'], output_dict=True)
            
            # Create metrics dataframe
            metrics_df = pd.DataFrame(report).transpose()
            st.dataframe(metrics_df.round(4))
            
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig = plot_confusion_matrix(cm, ['Negative', 'Positive'])
            st.plotly_chart(fig, use_container_width=True)
            
            # Prediction distribution
            st.subheader("Prediction Confidence Distribution")
            fig = px.histogram(
                predictions.flatten(),
                nbins=50,
                title="Distribution of Prediction Probabilities",
                labels={'x': 'Prediction Probability', 'y': 'Count'}
            )
            fig.add_vline(x=0.5, line_dash="dash", line_color="red", 
                         annotation_text="Decision Threshold")
            st.plotly_chart(fig, use_container_width=True)
            
            # Sample predictions
            st.subheader("Sample Predictions")
            sample_indices = np.random.choice(len(x_test), 5, replace=False)
            
            for idx in sample_indices:
                with st.expander(f"Sample {idx + 1}"):
                    actual = "Positive" if y_test[idx] == 1 else "Negative"
                    predicted = "Positive" if y_pred[idx] == 1 else "Negative"
                    confidence = predictions[idx][0]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**Actual:** {actual}")
                    with col2:
                        st.write(f"**Predicted:** {predicted}")
                    with col3:
                        st.write(f"**Confidence:** {confidence:.4f}")
                    
                    # Decode and display review
                    if st.session_state.word_index:
                        review = decode_review(x_test[idx], st.session_state.word_index)
                        st.write(f"**Review:** {review[:500]}...")
                        
        except Exception as e:
            st.error(f"Error during evaluation: {str(e)}")

def live_prediction_page():
    st.header("🔮 Live Prediction")
    
    if st.session_state.model is None:
        st.warning("Please train a model first in the 'Model Training' page.")
        return
    
    model = st.session_state.model
    word_index = st.session_state.word_index
    
    if word_index is None:
        st.warning("Word index not available. Please reload the dataset in 'Data Overview' page.")
        return
    
    st.subheader("Enter a movie review for sentiment analysis:")
    
    # Sample reviews for quick testing
    sample_reviews = {
        "Positive Sample": "This movie is absolutely fantastic! The acting was superb, the plot was engaging, and I was entertained throughout. Highly recommended!",
        "Negative Sample": "This was a terrible movie. The plot made no sense, the acting was awful, and I regretted watching it. Complete waste of time.",
        "Neutral Sample": "The movie was okay. Some parts were good, others not so much. It's an average film that you might enjoy if you have nothing else to watch."
    }
    
    col1, col2 = st.columns(2)
    with col1:
        selected_sample = st.selectbox("Quick Test Samples:", ["Custom"] + list(sample_reviews.keys()))
    
    if selected_sample != "Custom":
        user_input = st.text_area("Review Text:", value=sample_reviews[selected_sample], height=150)
    else:
        user_input = st.text_area("Review Text:", height=150, placeholder="Enter your movie review here...")
    
    if st.button("Analyze Sentiment", type="primary"):
        if user_input.strip():
            with st.spinner("Analyzing sentiment..."):
                try:
                    # Preprocess the input
                    processed_input = preprocess_text_for_prediction(user_input, word_index, maxlen=500)
                    
                    # Make prediction
                    prediction = model.predict(processed_input)
                    confidence = prediction[0][0]
                    
                    # Display results
                    st.subheader("Prediction Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    if confidence > 0.5:
                        sentiment = "Positive 😊"
                        color = "green"
                    else:
                        sentiment = "Negative 😔"
                        color = "red"
                        confidence = 1 - confidence
                    
                    with col1:
                        st.metric("Predicted Sentiment", sentiment)
                    with col2:
                        st.metric("Confidence", f"{confidence:.4f}")
                    with col3:
                        st.metric("Raw Score", f"{prediction[0][0]:.4f}")
                    
                    # Confidence visualization
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=confidence,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Confidence Level"},
                        gauge={
                            'axis': {'range': [None, 1]},
                            'bar': {'color': color},
                            'steps': [
                                {'range': [0, 0.5], 'color': "lightgray"},
                                {'range': [0.5, 1], 'color': "gray"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 0.8
                            }
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
        else:
            st.warning("Please enter a review to analyze.")

def preprocess_text_for_prediction(text, word_index, maxlen=500):
    """Preprocess text for model prediction"""
    # Simple tokenization and conversion to indices
    words = text.lower().split()
    sequence = []
    
    for word in words:
        if word in word_index:
            if word_index[word] < 10000:  # Only use words in our vocabulary
                sequence.append(word_index[word])
    
    # Pad sequence
    sequence = pad_sequences([sequence], maxlen=maxlen)
    return sequence

def model_architecture_page():
    st.header("🏗️ Model Architecture")
    
    if st.session_state.model is None:
        st.warning("Please train a model first in the 'Model Training' page.")
        return
    
    model = st.session_state.model
    
    # Model Summary
    st.subheader("Model Summary")
    
    # Capture model summary
    import io
    import sys
    
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    model.summary()
    summary_string = buffer.getvalue()
    sys.stdout = old_stdout
    
    st.text(summary_string)
    
    # Model Parameters
    st.subheader("Model Parameters")
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Parameters", f"{total_params:,}")
    with col2:
        st.metric("Trainable Parameters", f"{trainable_params:,}")
    with col3:
        st.metric("Non-trainable Parameters", f"{non_trainable_params:,}")
    
    # Layer Information
    st.subheader("Layer Details")
    layer_data = []
    for i, layer in enumerate(model.layers):
        layer_info = {
            "Layer #": i + 1,
            "Name": layer.name,
            "Type": type(layer).__name__,
            "Output Shape": str(layer.output_shape) if hasattr(layer, 'output_shape') else "N/A",
            "Params": layer.count_params()
        }
        layer_data.append(layer_info)
    
    layer_df = pd.DataFrame(layer_data)
    st.dataframe(layer_df, use_container_width=True)
    
    # Training History (if available)
    if st.session_state.history is not None:
        st.subheader("Training History")
        history = st.session_state.history
        
        # Create training history plot
        fig = plot_training_history(history)
        st.plotly_chart(fig, use_container_width=True)
        
        # Best metrics
        st.subheader("Best Training Metrics")
        best_train_acc = max(history.history['accuracy'])
        best_val_acc = max(history.history['val_accuracy'])
        best_train_loss = min(history.history['loss'])
        best_val_loss = min(history.history['val_loss'])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Best Training Accuracy", f"{best_train_acc:.4f}")
        with col2:
            st.metric("Best Validation Accuracy", f"{best_val_acc:.4f}")
        with col3:
            st.metric("Best Training Loss", f"{best_train_loss:.4f}")
        with col4:
            st.metric("Best Validation Loss", f"{best_val_loss:.4f}")

if __name__ == "__main__":
    main()

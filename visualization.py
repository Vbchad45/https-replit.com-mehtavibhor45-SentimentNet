import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

def plot_training_history(history):
    """
    Plot training history with accuracy and loss.
    
    Args:
        history: Keras training history object
        
    Returns:
        Plotly figure
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Model Accuracy', 'Model Loss'),
        vertical_spacing=0.1
    )
    
    epochs = range(1, len(history.history['accuracy']) + 1)
    
    # Add accuracy traces
    fig.add_trace(
        go.Scatter(
            x=list(epochs),
            y=history.history['accuracy'],
            mode='lines+markers',
            name='Training Accuracy',
            line=dict(color='#1f77b4')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=list(epochs),
            y=history.history['val_accuracy'],
            mode='lines+markers',
            name='Validation Accuracy',
            line=dict(color='#ff7f0e')
        ),
        row=1, col=1
    )
    
    # Add loss traces
    fig.add_trace(
        go.Scatter(
            x=list(epochs),
            y=history.history['loss'],
            mode='lines+markers',
            name='Training Loss',
            line=dict(color='#1f77b4'),
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=list(epochs),
            y=history.history['val_loss'],
            mode='lines+markers',
            name='Validation Loss',
            line=dict(color='#ff7f0e'),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_yaxes(title_text="Accuracy", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=2, col=1)
    
    fig.update_layout(
        height=600,
        title_text="Training History",
        hovermode='x unified'
    )
    
    return fig

def plot_confusion_matrix(cm, class_names):
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        cm: Confusion matrix (numpy array)
        class_names: List of class names
        
    Returns:
        Plotly figure
    """
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create annotations
    annotations = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            annotations.append(
                dict(
                    x=j, y=i,
                    text=f"{cm[i][j]}<br>({cm_normalized[i][j]:.2%})",
                    showarrow=False,
                    font=dict(color="white" if cm_normalized[i][j] > 0.5 else "black")
                )
            )
    
    fig = go.Figure(data=go.Heatmap(
        z=cm_normalized,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        showscale=True,
        colorbar=dict(title="Normalized Count")
    ))
    
    fig.update_layout(
        title="Confusion Matrix",
        xaxis=dict(title="Predicted Label"),
        yaxis=dict(title="True Label"),
        annotations=annotations,
        width=500,
        height=400
    )
    
    return fig

def plot_word_distribution(word_freq_dict, top_n=20):
    """
    Plot word frequency distribution.
    
    Args:
        word_freq_dict: Dictionary with word frequencies
        top_n: Number of top words to display
        
    Returns:
        Plotly figure
    """
    # Get top N words
    top_words = list(word_freq_dict.keys())[:top_n]
    top_counts = list(word_freq_dict.values())[:top_n]
    
    fig = px.bar(
        x=top_counts,
        y=top_words,
        orientation='h',
        title=f'Top {top_n} Most Frequent Words',
        labels={'x': 'Frequency', 'y': 'Words'}
    )
    
    fig.update_layout(
        height=600,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def plot_sequence_length_distribution(lengths, title="Sequence Length Distribution"):
    """
    Plot distribution of sequence lengths.
    
    Args:
        lengths: List of sequence lengths
        title: Plot title
        
    Returns:
        Plotly figure
    """
    fig = px.histogram(
        x=lengths,
        nbins=50,
        title=title,
        labels={'x': 'Sequence Length', 'y': 'Count'}
    )
    
    # Add statistics as vertical lines
    mean_length = np.mean(lengths)
    median_length = np.median(lengths)
    
    fig.add_vline(
        x=mean_length,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_length:.1f}"
    )
    
    fig.add_vline(
        x=median_length,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Median: {median_length:.1f}"
    )
    
    return fig

def plot_class_distribution(y_train, y_test=None, class_names=['Negative', 'Positive']):
    """
    Plot class distribution for training and test sets.
    
    Args:
        y_train: Training labels
        y_test: Test labels (optional)
        class_names: List of class names
        
    Returns:
        Plotly figure
    """
    if y_test is not None:
        # Create subplots for both train and test
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Training Set', 'Test Set'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Training set distribution
        train_counts = [np.sum(y_train == 0), np.sum(y_train == 1)]
        fig.add_trace(
            go.Bar(x=class_names, y=train_counts, name='Training', 
                   marker_color=['#ff6b6b', '#4ecdc4']),
            row=1, col=1
        )
        
        # Test set distribution
        test_counts = [np.sum(y_test == 0), np.sum(y_test == 1)]
        fig.add_trace(
            go.Bar(x=class_names, y=test_counts, name='Test', 
                   marker_color=['#ff6b6b', '#4ecdc4'], showlegend=False),
            row=1, col=2
        )
        
        fig.update_layout(title="Class Distribution")
        
    else:
        # Single plot for training set only
        train_counts = [np.sum(y_train == 0), np.sum(y_train == 1)]
        fig = px.bar(
            x=class_names,
            y=train_counts,
            title="Class Distribution",
            color=class_names,
            color_discrete_map={'Negative': '#ff6b6b', 'Positive': '#4ecdc4'}
        )
    
    return fig

def plot_prediction_confidence(predictions, threshold=0.5):
    """
    Plot distribution of prediction confidence scores.
    
    Args:
        predictions: Array of prediction probabilities
        threshold: Decision threshold
        
    Returns:
        Plotly figure
    """
    fig = px.histogram(
        predictions.flatten(),
        nbins=50,
        title="Distribution of Prediction Confidence",
        labels={'x': 'Prediction Probability', 'y': 'Count'}
    )
    
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold: {threshold}"
    )
    
    # Add statistics
    mean_conf = np.mean(predictions)
    fig.add_vline(
        x=mean_conf,
        line_dash="dot",
        line_color="blue",
        annotation_text=f"Mean: {mean_conf:.3f}"
    )
    
    return fig

def plot_metrics_comparison(metrics_dict):
    """
    Plot comparison of different metrics.
    
    Args:
        metrics_dict: Dictionary with metric names as keys and values as values
        
    Returns:
        Plotly figure
    """
    metrics = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    fig = px.bar(
        x=metrics,
        y=values,
        title="Model Performance Metrics",
        labels={'x': 'Metrics', 'y': 'Score'}
    )
    
    fig.update_layout(
        xaxis={'categoryorder': 'total descending'},
        yaxis={'range': [0, 1]}
    )
    
    return fig

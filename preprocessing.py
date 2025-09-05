import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import string

def preprocess_data(x_train, x_test, maxlen=500):
    """
    Preprocess training and test data by padding sequences.
    
    Args:
        x_train: Training sequences
        x_test: Test sequences  
        maxlen: Maximum length for padding
        
    Returns:
        Padded training and test sequences
    """
    x_train_padded = pad_sequences(x_train, maxlen=maxlen)
    x_test_padded = pad_sequences(x_test, maxlen=maxlen)
    
    return x_train_padded, x_test_padded

def decode_review(encoded_review, word_index, reverse_word_index=None):
    """
    Decode an encoded review back to text.
    
    Args:
        encoded_review: List of word indices
        word_index: Dictionary mapping words to indices
        reverse_word_index: Dictionary mapping indices to words (optional)
        
    Returns:
        Decoded review as string
    """
    if reverse_word_index is None:
        # Create reverse word index
        reverse_word_index = {value: key for key, value in word_index.items()}
    
    # Decode the review
    # Note: indices are offset by 3 because 0, 1, 2 are reserved for padding, start, unknown
    decoded_words = []
    for index in encoded_review:
        if index >= 3:
            decoded_words.append(reverse_word_index.get(index - 3, '<UNK>'))
        elif index == 2:
            decoded_words.append('<START>')
        elif index == 1:
            decoded_words.append('<UNK>')
        else:  # index == 0
            decoded_words.append('<PAD>')
    
    return ' '.join(decoded_words)

def clean_text(text):
    """
    Clean text by removing special characters, extra whitespace, etc.
    
    Args:
        text: Input text string
        
    Returns:
        Cleaned text string
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text

def text_to_sequences(texts, word_index, maxlen=500, num_words=10000):
    """
    Convert text to sequences of word indices.
    
    Args:
        texts: List of text strings
        word_index: Dictionary mapping words to indices
        maxlen: Maximum sequence length
        num_words: Maximum number of words to keep
        
    Returns:
        Padded sequences
    """
    sequences = []
    
    for text in texts:
        # Clean and tokenize text
        cleaned_text = clean_text(text)
        words = cleaned_text.split()
        
        # Convert words to indices
        sequence = []
        for word in words:
            if word in word_index and word_index[word] < num_words:
                sequence.append(word_index[word])
            else:
                sequence.append(1)  # Unknown token
        
        sequences.append(sequence)
    
    # Pad sequences
    padded_sequences = pad_sequences(sequences, maxlen=maxlen)
    
    return padded_sequences

def get_sequence_stats(sequences):
    """
    Get statistics about sequence lengths.
    
    Args:
        sequences: List of sequences
        
    Returns:
        Dictionary with statistics
    """
    lengths = [len(seq) for seq in sequences]
    
    stats = {
        'min_length': np.min(lengths),
        'max_length': np.max(lengths),
        'mean_length': np.mean(lengths),
        'median_length': np.median(lengths),
        'std_length': np.std(lengths),
        'percentile_95': np.percentile(lengths, 95),
        'percentile_99': np.percentile(lengths, 99)
    }
    
    return stats

def create_word_frequency_dict(sequences, word_index):
    """
    Create a frequency dictionary for words in sequences.
    
    Args:
        sequences: List of sequences (word indices)
        word_index: Dictionary mapping words to indices
        
    Returns:
        Dictionary with word frequencies
    """
    # Create reverse mapping
    index_to_word = {idx: word for word, idx in word_index.items()}
    
    # Count word frequencies
    word_freq = {}
    
    for sequence in sequences:
        for idx in sequence:
            if idx in index_to_word:
                word = index_to_word[idx]
                word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency
    sorted_freq = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True))
    
    return sorted_freq

def remove_stopwords(text, stopwords=None):
    """
    Remove stopwords from text.
    
    Args:
        text: Input text string
        stopwords: List of stopwords (optional)
        
    Returns:
        Text with stopwords removed
    """
    if stopwords is None:
        # Basic English stopwords
        stopwords = set([
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'this', 'that',
            'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours',
            'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he',
            'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
            'itself', 'they', 'them', 'their', 'theirs', 'themselves'
        ])
    
    words = text.lower().split()
    filtered_words = [word for word in words if word not in stopwords]
    
    return ' '.join(filtered_words)

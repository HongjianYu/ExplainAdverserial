// src/components/ImageSelection.js
import React from 'react';
import Modal from 'react-modal';

Modal.setAppElement('#root');

function ImageSelection({ isOpen, imageId, onRequestClose, mode }) {
    let imagePath;
    if (mode === 'Top-K') {
        imagePath = 'topk_results';
    } else if (mode === 'Filter') {
        imagePath = 'filter_results';
    } else if (mode === 'Aggregation') {
        imagePath = 'aggregation_results';
    }
    const imageUrl = `http://localhost:8000/${imagePath}/${imageId}.jpg`;

    const handleImageLoadError = () => {
        console.error('Failed to load image with ID:', imageId);
    };

    return (
        <Modal 
            isOpen={isOpen} 
            onRequestClose={onRequestClose} 
            style={{
                overlay: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)' // Optional: dark overlay
                },
                content: {
                    position: 'fixed',
                    top: '50%',
                    left: '50%',
                    right: 'auto',
                    bottom: 'auto',
                    transform: 'translate(-50%, -50%)',
                    maxWidth: '90%', // Limiting image size
                    maxHeight: '90%', // Limiting image size
                    overflow: 'auto' // Ensures content can be scrolled if larger than the modal
                }
            }}
        >
            <div style={{ textAlign: 'center' }}>
                <img
                    src={imageUrl}
                    alt={`Selected Image ${imageId}`}
                    onError={handleImageLoadError}
                    style={{ maxWidth: '100%', maxHeight: '80vh' }} // Resizes image to not be too large
                />
                <div>
                    <button onClick={onRequestClose} style={{ marginTop: '20px' }}>Close</button>
                </div>
            </div>
        </Modal>
    );
}

export default ImageSelection;

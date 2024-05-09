// src/components/ImageSelection.js
import React, { useState } from 'react';
import Modal from 'react-modal';

Modal.setAppElement('#root');

function ImageSelection({ isOpen, imageId, onRequestClose, mode }) {
    const [stat, setStat] = useState('');

    const imageUrl = `http://localhost:8000/topk_results/${imageId}.JPEG`;
    const statUrl = `http://localhost:8000/topk_labels/${imageId}`

    const handleImageLoadError = () => {
        console.error('Failed to load image with ID:', imageId);
    };

    const fetchImageLabels = async (path) => {
        return (await fetch(path, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            },
        })).json();
    };

    const handleImageLabels = async () => {
        const labels = await fetchImageLabels(statUrl)
        const correctness = labels.correctness ? 'Correct' : 'Incorrect'
        const attack = labels.attack ? 'Yes' : 'No'
        setStat(`Prediction: ${correctness};  Attack: ${attack}`);
    };

    handleImageLabels()

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
                <div style={{ marginTop: '5px' }}>
                    {stat}
                </div>
                <div>
                    <button onClick={onRequestClose} style={{ marginTop: '5px' }}>Close</button>
                </div>
            </div>
        </Modal>
    );
}

export default ImageSelection;

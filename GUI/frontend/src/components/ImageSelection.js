// src/components/ImageSelection.js
import React from 'react';
import Modal from 'react-modal';

Modal.setAppElement('#root');

function ImageSelection({ isOpen, imageId, onRequestClose }) {
    return (
        <Modal isOpen={isOpen} onRequestClose={onRequestClose}>
            <img
                src={`http://localhost:8000/images/${imageId}`}
                alt={`Selected Image ${imageId}`}
            />
            <button onClick={onRequestClose}>Close</button>
        </Modal>
    );
}

export default ImageSelection;

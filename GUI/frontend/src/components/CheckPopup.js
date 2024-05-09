import React from 'react';
import './CheckPopup.css';

const CheckPopup = ({ isOpen, onClose }) => {
    if (!isOpen) return null;

    // Dummy data showing 25 images
    const images = Array(25).fill('1.jpg');

    return (
        <div className="check-popup-overlay">
            <div className="check-popup">
                <button className="close-btn" onClick={onClose}>x</button>
                <div className="images-grid">
                    {images.map((img, index) => (
                        <img key={index} src={`http://localhost:8000/topk_results/${img}`} alt={`Image ${index}`} className="popup-img" />
                    ))}
                </div>
            </div>
        </div>
    );
};

export default CheckPopup;

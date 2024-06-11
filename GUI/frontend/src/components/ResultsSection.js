// src/components/ResultsSection.js
import React from 'react';
import './ResultsSection.css';

// TODO: Add the field for augment
function ResultsSection({ imageIds, mode, onSelectImage, executionTime, skippedImages }) {
    console.log('ResultsSection mode:', mode);
    let imagePath;
    // TODO: Add another check for augment or noth
    if (mode === 'Top-K') {
        imagePath = 'topk_cams';
    }

    return (
            <div className="results-section">
                <div className="info-box">
                    <div className="execution-info">
                        <span className="time-label">Execution Time:</span>
                        <span className="time-value">{executionTime.toFixed(3)} seconds</span>
                    </div>
                    <div className="image-count-info">
                        <span className="image-count-label">Skipped Images Count:</span>
                        <span className="image-count-value">{skippedImages}</span>
                    </div>
                </div>
                <div className="image-container">
                    {imageIds.map((id) => (
                        <img
                            key={id}
                            src={`http://localhost:8000/${imagePath}/${id}.JPEG`}
                            alt={`Image ${id}`}
                            onClick={() => onSelectImage(id)}
                        />
                    ))}
                </div>
            </div>
    );
}

export default ResultsSection;

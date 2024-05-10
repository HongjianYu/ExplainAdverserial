// src/components/ResultsSection.js
import React from 'react';
import './ResultsSection.css';

// TODO: Add the field for augment
function ResultsSection({ imageIds, mode, onSelectImage }) {
    console.log('ResultsSection mode:', mode);
    let imagePath;
    // TODO: Add another check for augment or noth
    if (mode === 'Top-K') {
        imagePath = 'topk_cams';
    }

    return (
            <div className="results-section">
                {imageIds.map((id) => (
                    <img
                        key={id}
                        src={`http://localhost:8000/${imagePath}/${id}.JPEG`}
                        alt={`Image ${id}`}
                        onClick={() => onSelectImage(id)}
                    />
                ))}
            </div>
    );
}

export default ResultsSection;

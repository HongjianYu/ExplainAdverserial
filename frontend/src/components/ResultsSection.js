// src/components/ResultsSection.js
import React from 'react';
import './ResultsSection.css'; // Assuming you have styling defined in ResultsSection.css

function ResultsSection({ imageIds, mode, onSelectImage }) {
    console.log('ResultsSection mode:', mode);
    const imagePath = mode === 'Top-K' ? 'topk_results' : 'filter_results';

    return (
        <div className="results-section">
            {imageIds.map((id) => (
                <img
                    key={id}
                    src={`http://localhost:5000/${imagePath}/${id}.jpg`}
                    alt={`Image ${id}`}
                    onClick={() => onSelectImage(id)}
                />
            ))}
        </div>
    );
}

export default ResultsSection;

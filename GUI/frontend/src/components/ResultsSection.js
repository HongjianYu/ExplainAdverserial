// src/components/ResultsSection.js
import React from 'react';
import './ResultsSection.css';

function ResultsSection({ imageIds, mode, onSelectImage }) {
    console.log('ResultsSection mode:', mode);
    let imagePath;
    if (mode === 'Top-K') {
        imagePath = 'topk_results';
    } else if (mode === 'Filter') {
        imagePath = 'filter_results';
    } else if (mode === 'Aggregation') {
        imagePath = 'aggregate_results';
    }

    return (
        <div className="results-section">
            {imageIds.map((id) => (
                <img
                    key={id}
                    src={`http://localhost:8000/${imagePath}/${id}.jpg`}
                    alt={`Image ${id}`}
                    onClick={() => onSelectImage(id)}
                />
            ))}
        </div>
    );
}

export default ResultsSection;

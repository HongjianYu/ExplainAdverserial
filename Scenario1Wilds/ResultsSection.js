// src/components/ResultsSection.js
import React from 'react';
import './ResultsSection.css'; // Assuming you have styling defined in ResultsSection.css

function ResultsSection({ imageIds, mode, onSelectImage, aug, executionTime}) {
    console.log('ResultsSection mode:', mode);
    
    let imagePath;
    if (mode === 'Top-K') {
        imagePath = 'topk_results';
    } else if (mode === 'Filter') {
        imagePath = 'topk_results';
    } else if (mode === 'Aggregation') {
        imagePath = 'aggregation_results';
    }
    if (aug == true){
        imagePath = 'augment_results'
    }
    let len;
    len = imageIds.length
    if (imageIds.length > 50){
        imageIds = imageIds.slice(0, 50);
    }
    // return (
    //     <div className="results-section">
    //         {imageIds.map((id) => (
    //             <img
    //                 key={id}
    //                 src={`http://localhost:8000/${imagePath}/${id}.png`}
    //                 alt={`Image ${id}`}
    //                 onClick={() => onSelectImage(id)}
    //             />
    //         ))}
    //     </div>
    // );
    return (
        <div className="results-section">
            <div className="info-box">
                <div className="execution-info">
                    <span className="time-label">Execution Time:</span>
                    <span className="time-value">{executionTime.toFixed(3)} seconds</span>
                </div>
                <div className="image-count-info">
                    <span className="image-count-label">Returned Examples:</span>
                    <span className="image-count-value">{len}</span>
                </div>
            </div>
            <div className="image-container">
                {imageIds.map((id) => (
                <img
                    key={id}
                    src={`http://localhost:9000/${imagePath}/${id}.png`}
                    alt={`Image ${id}`}
                    onClick={() => onSelectImage(id)}
                />
            ))}
            </div>
            
        </div>
    );
}

export default ResultsSection;



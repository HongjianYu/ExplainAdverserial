import React from 'react';
import './ResultsSection.css'; // Assuming you have styling defined in ResultsSection.css

function ResultsSection({ imageIds, mode, onSelectImage, aug }) {
    console.log('ResultsSection mode:', mode);
    
    let imagePath_sal = "saliency_images";
    let imagePath_att = "human_att_images";
    console.log(imageIds);

    return (
        <div className="results-section">
            {imageIds.map((id) => (
                <div key={id} className="image-wrapper">
                    <img
                        src={`http://localhost:8000/${imagePath_sal}/${id}_saliency.jpg`}
                        alt={`Saliency Image ${id}`}
                        onClick={() => onSelectImage(id)}
                    />
                    <img
                        src={`http://localhost:8000/${imagePath_att}/${id}.jpg`}
                        alt={`Human Attention Image ${id}`}
                        onClick={() => onSelectImage(id)}
                    />
                </div>
            ))}
        </div>
    );
}

export default ResultsSection;
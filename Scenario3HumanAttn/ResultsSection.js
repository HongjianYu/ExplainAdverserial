import React from 'react';
import './ResultsSection.css'; // Assuming you have styling defined in ResultsSection.css

function ResultsSection({ imageIds, mode, onSelectImage, executionTime, imagesCount, aug }) {
  console.log('ResultsSection mode:', mode);

  // Use fixed paths for saliency, human attention, intersect, and union images
  let imagePath_sal = 'saliency_images';
  let imagePath_att = 'human_att_images';
  let imagePath_int = 'intersect_visualization';
  let imagePath_uni = 'union_visualization';

  console.log(imageIds);

  return (
    <div className="results-section">
      <div className="info-box">
        <div className="execution-info">
          <span className="time-label">Execution Time:</span>
          <span className="time-value">{executionTime} seconds</span>
        </div>
        <div className="image-count-info">
          <span className="image-count-label">Images:</span>
          <span className="image-count-value">{imagesCount}</span>
        </div>
      </div>
      <div className="image-container">
        {imageIds.map((id) => (
          <div key={id} className="image-row">
            <div className="image-wrapper">
              <img
                src={`http://localhost:8080/${imagePath_sal}/${id}_saliency.jpg`}
                alt={`Saliency Image ${id}`}
                onClick={() => onSelectImage(id)}
              />
              <p>Saliency Mask {id}</p>
            </div>
            <div className="image-wrapper">
              <img
                src={`http://localhost:8080/${imagePath_att}/${id}.jpg`}
                alt={`Human Attention Image ${id}`}
                onClick={() => onSelectImage(id)}
              />
              <p>Human Attention Mask {id}</p>
            </div>
            <div className="image-wrapper">
              <img
                src={`http://localhost:8080/${imagePath_int}/intersect_result_${id}.png`}
                alt={`Intersect Image ${id}`}
                onClick={() => onSelectImage(id)}
              />
              <p>Intersection Mask {id}</p>
            </div>
            <div className="image-wrapper">
              <img
                src={`http://localhost:8080/${imagePath_uni}/union_result_${id}.png`}
                alt={`Union Image ${id}`}
                onClick={() => onSelectImage(id)}
              />
              <p>Union Mask {id}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default ResultsSection;

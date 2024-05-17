// ITERATION WORKS
// src/App.js
import React, { useState } from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import InputSection from './components/InputSection';
import ResultsSection from './components/ResultsSection';
import ImageSelection from './components/ImageSelection';
import DataPreparation from './components/DataPreparation';

import './App.css';

function App() {
  const [imageIds, setImageIds] = useState([]);
  const [selectedImageId, setSelectedImageId] = useState(null);
  const [mode, setMode] = useState('Top-K');
  const [isAugment, setAugment] = useState(false);
  const [executionTime, setExecutionTime] = useState(0);
  const [imagesCount, setImagesCount] = useState(0); // Add this line

  const handleSearchResults = (results) => {
    setImageIds(results);
  };

  const handleModeChange = (newMode) => {
    setMode(newMode);
  };

  const handleImageClick = (imageId) => {
    setSelectedImageId(imageId);
  };

  const closeImageSelection = () => {
    setSelectedImageId(null);
  };

  const handleStartAugment = (status) => {
    setAugment(status);
  };

  return (
    <Router>
      <div className="app">
        <header className="app-header">
          MaskSearch
        </header>
        <Routes>
          <Route path="/data-preparation" element={<DataPreparation />} />
          <Route path="/input" element={
            <div className="main-content">
              <InputSection 
                onSearchResults={handleSearchResults}
                onModeChange={handleModeChange}
                setExecutionTime={setExecutionTime} // Pass setExecutionTime
                setImagesCount={setImagesCount} // Pass setImagesCount
              />
              <ResultsSection 
                imageIds={imageIds}
                onSelectImage={handleImageClick} 
                mode={mode}
                executionTime={executionTime} // Pass executionTime
                imagesCount={imagesCount} // Pass imagesCount
              />
              {selectedImageId && (
                  <ImageSelection
                      isOpen={!!selectedImageId}
                      imageId={selectedImageId}
                      onRequestClose={closeImageSelection}
                      mode={mode}
                  />
              )}
            </div>
          } />
          <Route path="/" element={<DataPreparation />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;

// src/App.js
import React, { useState } from 'react';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import InputSection from './components/InputSection';
import ResultsSection from './components/ResultsSection';
import ImageSelection from './components/ImageSelection';
import DataPreparation from './components/DataPreparation';

import './App.css';

function App() {
  const [imageIds, setImageIds] = useState([]);
  const [selectedImageId, setSelectedImageId] = useState(null);
  const [mode, setMode] = useState('Top-K');
  const [toggle, setEnable] = useState(true);
  const [executionTime, setExecutionTime] = useState(0);
  const [skippedImages, setSkippedImages] = useState(0);

  // Handle the search results from InputSection
  const handleSearchResults = (results) => {
    setImageIds(results); // Assuming results is an array of image IDs
  };

  // Function to update mode from InputSection
  const handleModeChange = (newMode) => {
    setMode(newMode); // Update the mode based on user selection
  };

  // Handle the selected image from ResultsSection
  const handleImageClick = (imageId) => {
    setSelectedImageId(imageId);
  };

  // Close the ImageSelection modal
  const closeImageSelection = () => {
    setSelectedImageId(null);
  };

  // Toggle the state when the slider is clicked
  const handleToggle = () => {
    setEnable(!toggle);
  };

  return (
    <Router>
      <div className="app">
        <header className="app-header">
            MaskSearch - Searching for Adversarial Attacks
        </header>
        <label class="switch">
            <input type="checkbox" checked={toggle} onChange={handleToggle} />
            <span class="slider round"></span>
            <span class='slider-label'>Enable MaskSearch</span>
        </label>
        <Routes>
          <Route path="/data-preparation" element={<DataPreparation />} />
          <Route path="/input" element={
            <div className="main-content">
              {/* TODO: add a field for augment in both InputSection and ResultsSection */}
              <InputSection
                  onSearchResults={handleSearchResults}
                  onModeChange={handleModeChange}
                  setExecutionTime={setExecutionTime}
                  setSkippedImages={setSkippedImages}
                  ms={toggle}
              />
              <ResultsSection
                  imageIds={imageIds}
                  onSelectImage={handleImageClick}
                  mode={mode}
                  executionTime={executionTime}
                  skippedImages={skippedImages}
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
          {/* Redirect from home to /data-preparation */}
          <Route path="/" element={<DataPreparation />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;

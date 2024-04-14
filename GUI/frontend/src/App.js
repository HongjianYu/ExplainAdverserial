// src/App.js
import React, { useState } from 'react';
import InputSection from './components/InputSection';
import ResultsSection from './components/ResultsSection';
import ImageSelection from './components/ImageSelection';
import './App.css';

function App() {
  const [imageIds, setImageIds] = useState([]);
  const [selectedImageId, setSelectedImageId] = useState(null);
  const [mode, setMode] = useState('Top-K');

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

  return (
      <div className="app">
        <header className="app-header">MaskSearch</header>
        <div className="main-content">
          <InputSection onSearchResults={handleSearchResults} onModeChange={handleModeChange} />
          <ResultsSection imageIds={imageIds} onSelectImage={handleImageClick} mode={mode} />
        </div>
        {selectedImageId && (
            <ImageSelection
                isOpen={!!selectedImageId}
                imageId={selectedImageId}
                onRequestClose={closeImageSelection}
            />
        )}
      </div>
  );
}

export default App;

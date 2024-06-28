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
  const [isAugment, setAugment] = useState(false);
  const [toggle, setEnable] = useState(false);
  const [executionTime, setExecutionTime] = useState(0);
  const [imagesCount, setImagesCount] = useState(0);
  const [skippedImages, setSkippedImages] = useState(0);
  const [queryPerformed, setQueryPerformed] = useState(false);

  // Handle the search results from InputSection
  const handleSearchResults = (results) => {
    setImageIds(results);
    setQueryPerformed(true);
  };

  // Function to update mode from InputSection
  const handleModeChange = (newMode) => {
    setMode(newMode);
    setQueryPerformed(false);
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

  const handleStartAugment = (status) => {
    setAugment(status);
  };

  return (
    <Router>
      <div className="app">
        <header className="app-header">
          <div className="header-title">
            MaskSearch
          </div>
          <nav>
            <Link to="/scenario1">Scenario 1</Link>
            <Link to="/scenario2">Scenario 2</Link>
            <Link to="/scenario3">Scenario 3</Link>
          </nav>
        </header>
        <div className="toggle-container">
          <label className="switch">
            <input type="checkbox" checked={toggle} onChange={handleToggle} />
            <span className="slider round"></span>
          </label>
          <span className='slider-label'>Toggle to Enable MaskSearch</span>
        </div>
        <Routes>
          <Route path="/data-preparation" element={<DataPreparation />} />
          <Route path="/scenario1" element={
            <div className="main-content">
              <InputSection 
                scenario="scenario1"
                onSearchResults={handleSearchResults}
                onModeChange={handleModeChange}
                setExecutionTime={setExecutionTime}
                isAug={handleStartAugment}
                ms={toggle}
              />
              {queryPerformed && (
                <ResultsSection 
                  scenario="scenario1"
                  imageIds={imageIds}
                  onSelectImage={handleImageClick} 
                  mode={mode}
                  aug={isAugment}
                  executionTime={executionTime}
                />
              )}
              {selectedImageId && (
                <ImageSelection
                  scenario="scenario1"
                  isOpen={!!selectedImageId}
                  imageId={selectedImageId}
                  onRequestClose={closeImageSelection}
                  mode={mode}
                />
              )}
            </div>
          } />
          <Route path="/scenario2" element={
            <div className="main-content">
              <InputSection 
                scenario="scenario2"
                onSearchResults={handleSearchResults}
                onModeChange={handleModeChange}
                setExecutionTime={setExecutionTime}
                ms={toggle}
                setSkippedImages={setSkippedImages}
              />
              {queryPerformed && (
                <ResultsSection 
                  scenario="scenario2"
                  imageIds={imageIds}
                  onSelectImage={handleImageClick} 
                  mode={mode}
                  executionTime={executionTime}
                  skippedImages={skippedImages}
                />
              )}
              {selectedImageId && (
                <ImageSelection
                  scenario="scenario2"
                  isOpen={!!selectedImageId}
                  imageId={selectedImageId}
                  onRequestClose={closeImageSelection}
                  mode={mode}
                />
              )}
            </div>
          } />
          <Route path="/scenario3" element={
            <div className="main-content">
              <InputSection 
                scenario="scenario3"
                onSearchResults={handleSearchResults}
                onModeChange={handleModeChange}
                setExecutionTime={setExecutionTime}
                ms={toggle}
                setImagesCount={setImagesCount}
              />
              {queryPerformed && (
                <ResultsSection 
                  scenario="scenario3"
                  imageIds={imageIds}
                  onSelectImage={handleImageClick} 
                  mode={mode}
                  executionTime={executionTime}
                  imagesCount={imagesCount}
                />
              )}
              {selectedImageId && (
                <ImageSelection
                  scenario="scenario3"
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


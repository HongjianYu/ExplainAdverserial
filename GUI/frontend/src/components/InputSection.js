// src/components/InputSection.js
import React, { useState } from 'react';
import QueryCommand from './QueryCommand';
import './InputSection.css';

// TODO: Add the field for augment
function InputSection({ onSearchResults, onModeChange, setExecutionTime, setSkippedImages, ms }) {
    const [mode, setMode] = useState('Top-K');
    const [k, setK] = useState('20');
    const [pixelUpperBound, setPixelUpperBound] = useState('1.0');
    const [pixelLowerBound, setPixelLowerBound] = useState('0.0');
    const [order, setOrder] = useState('DESC');
    const [threshold, setThreshold] = useState('0.2');
    const [thresholdDirection, setThresholdDirection] = useState('>');
    const [queryCommand, setQueryCommand] = useState('');
    const [isQueryActive, setIsQueryActive] = useState(false);

    const fetchQueryCommand = async (path, body) => {
        const response = await fetch(`http://localhost:8000${path}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(body),
        });
        return response.json();
    };

    const handleStartQuery = async () => {
        let path, body;
        if (mode === 'Top-K') {
            path = '/api/topk_search';
            body = { k, pixelUpperBound, pixelLowerBound, order, ms };
        }

        const data = await fetchQueryCommand(path, body);
        setQueryCommand(data.query_command);
        onSearchResults(data.image_ids);
        setExecutionTime(data.execution_time);
        setSkippedImages(data.skipped_images_count);
        setIsQueryActive(true); // Enable the Augment button
    };

    const handleShowExecution = () => {
        // Implement later, make a window to show path
    };

    const handleModeChange = (newMode) => {
        setMode(newMode); // Update local state
        setIsQueryActive(false); // Reset the active state when changing modes
        onModeChange(newMode); // Inform the parent component about the mode change
        // TODO: Reset the status of augment to false when changing modes
    };

    return (
        <div className="input-section">
            <div className="header">
                <div className="mode-switch">
                    <button className={mode === 'Top-K' ? 'active' : ''} onClick={() => handleModeChange('Top-K')}>Top-K</button>
                    {/* <button className={mode === 'Filter' ? 'active' : ''} onClick={() => handleModeChange('Filter')}>Filter</button> */}
                    {/* <button className={mode === 'Aggregation' ? 'active' : ''} onClick={() => handleModeChange('Aggregation')}>Aggregation</button> */}
                </div>
            </div>
            {mode === 'Top-K' ? (
                <>
                    {/* Top-K specific fields */}
                    <div className="input-container">
                        <label htmlFor="K" className="input-label">K:</label>
                        <select id="k" value={k} onChange={(e) => setK(e.target.value)} className="input-field">
                            {[5, 10, 15, 20, 25].map(option => (
                                <option key={option} value={option}>{option}</option>
                            ))}
                        </select>
                    </div>
                    <div className="input-container">
                        <label htmlFor="ROI" className="input-label">ROI:</label>
                        <select id="roi" className="input-field" disabled>
                            {['full image'].map(option => (
                                <option key={option} value={option}>{option}</option>
                            ))}
                        </select>
                    </div>
                    <div className="input-container">
                        <label htmlFor="pixelUpperBound" className="input-label">Pixel Val Upper Bound:</label>
                        <input
                            id="pixelUpperBound"
                            type="text"
                            value={pixelUpperBound}
                            onChange={(e) => setPixelUpperBound(e.target.value)}
                            className="input-field"
                        />
                    </div>
                    <div className="input-container">
                        <label htmlFor="pixelLowerBound" className="input-label">Pixel Val Lower Bound:</label>
                        <input
                            id="pixelLowerBound"
                            type="text"
                            value={pixelLowerBound}
                            onChange={(e) => setPixelLowerBound(e.target.value)}
                            className="input-field"
                        />
                    </div>
                    <div className="input-container">
                        <label htmlFor="Order" className="input-label">Order:</label>
                        <select id="order" value={order} onChange={(e) => setOrder(e.target.value)} className="input-field">
                            {['ASC', 'DESC'].map(option => (
                                <option key={option} value={option}>{option}</option>
                            ))}
                        </select>
                    </div>
                </>
            ) : mode === 'Filter' ? (
                <>
                </>
            ) : (
                <>
                </>
            ) }
            <div className="halfsize-buttons">
                <button className="start-halfsize-btn" onClick={handleStartQuery}>
                    Start Query
                </button>
                {/* <button className="appending-halfsize-btn" onClick={handleShowExecution} disabled={!isQueryActive}>
                    Exec Details
                </button> */}
            </div>
            {/* <button onClick={() => {}} disabled={!isQueryActive}>Start Augment</button> */}
            <QueryCommand command={queryCommand} />
        </div>
    );
}

export default InputSection;

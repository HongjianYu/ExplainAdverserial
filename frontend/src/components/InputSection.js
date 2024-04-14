// src/components/InputSection.js
import React, { useState } from 'react';
import QueryCommand from './QueryCommand';
import './InputSection.css';

function InputSection({ onSearchResults, onModeChange }) {
    const [mode, setMode] = useState('Top-K');
    const [k, setK] = useState('5');
    const [roi, setRoi] = useState('object bounding box');
    const [pixelUpperBound, setPixelUpperBound] = useState('1');
    const [pixelLowerBound, setPixelLowerBound] = useState('0.5');
    const [order, setOrder] = useState('ASC');
    const [threshold, setThreshold] = useState('0.2');
    const [thresholdDirection, setThresholdDirection] = useState('>');
    const [queryCommand, setQueryCommand] = useState('');
    const [isQueryActive, setIsQueryActive] = useState(false);
    const [isPathActive, setIsPathActive] = useState(false);


    const fetchQueryCommand = async (path, body) => {
        const response = await fetch(`http://localhost:5000${path}`, {
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
            body = { k, roi, pixelUpperBound, pixelLowerBound, order };
        } else {
            path = '/api/filter_search';
            body = { threshold, thresholdDirection, roi, pixelUpperBound, pixelLowerBound };
        }

        const data = await fetchQueryCommand(path, body);
        setQueryCommand(data.query_command);
        onSearchResults(data.image_ids);
        setIsQueryActive(true); // Enable the Augment button
    };

    const handleShowExecution = () => {
        // Implement later, make a window to show path
    };

    const handleStartAugment = async () => {
        // const response = await fetch('http://localhost:5000/api/augment');
        // const data = await response.json();
        // console.log('Augment result:', data.result);
        setIsPathActive(true);
    };

    const handleShowPath = () => {
        // Implement later, make a window to show path
    };

    const handleModeChange = (newMode) => {
        setMode(newMode); // Update local state
        setIsQueryActive(false); // Reset the active state when changing modes
        onModeChange(newMode); // Inform the parent component about the mode change
    };

    return (
        <div className="input-section">
            <div className="header">
                <h2 className="title">{mode} Query</h2>
                <div className="mode-switch">
                    <button className={mode === 'Top-K' ? 'active' : ''} onClick={() => handleModeChange('Top-K')}>Top-K</button>
                    <button className={mode === 'Filter' ? 'active' : ''} onClick={() => handleModeChange('Filter')}>Filter</button>
                    <button className={mode === ''}>Aggregation</button>
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
                        <select id="roi" value={roi} onChange={(e) => setRoi(e.target.value)} className="input-field">
                            {['object bounding box', 'customizing box'].map(option => (
                                <option key={option} value={option}>{option}</option>
                            ))}
                        </select>
                    </div>
                    <div className="input-container">
                        <label htmlFor="pixelUpperBound" className="input-label">Pixel Value Upper Bound:</label>
                        <input
                            id="pixelUpperBound"
                            type="text"
                            value={pixelUpperBound}
                            onChange={(e) => setPixelUpperBound(e.target.value)}
                            className="input-field"
                        />
                    </div>
                    <div className="input-container">
                        <label htmlFor="pixelLowerBound" className="input-label">Pixel Value Lower Bound:</label>
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
            ) : (
                <>
                    {/* Filter specific fields */}
                    <div className="input-container">
                        <label htmlFor="threshold" className="input-label">Threshold:</label>
                        <div className="threshold-container">
                            <select
                                id="thresholdDirection"
                                value={thresholdDirection}
                                onChange={(e) => setThresholdDirection(e.target.value)}
                                className="threshold-field"
                            >
                                <option value=">">&gt;</option>
                                <option value="<">&lt;</option>
                            </select>
                            <input
                                id="thresholdValue"
                                type="text"
                                value={threshold}
                                onChange={(e) => setThreshold(e.target.value)}
                                className="threshold-field"
                            />
                        </div>
                    </div>
                    <div className="input-container">
                        <label htmlFor="ROI" className="input-label">ROI:</label>
                        <select id="roi" value={roi} onChange={(e) => setRoi(e.target.value)} className="input-field">
                            {['object bounding box', 'customizing box'].map(option => (
                                <option key={option} value={option}>{option}</option>
                            ))}
                        </select>
                    </div>
                    <div className="input-container">
                        <label htmlFor="pixelUpperBound" className="input-label">Pixel Value Upper Bound:</label>
                        <input
                            id="pixelUpperBound"
                            type="text"
                            value={pixelUpperBound}
                            onChange={(e) => setPixelUpperBound(e.target.value)}
                            className="input-field"
                        />
                    </div>
                    <div className="input-container">
                        <label htmlFor="pixelLowerBound" className="input-label">Pixel Value Lower Bound:</label>
                        <input
                            id="pixelLowerBound"
                            type="text"
                            value={pixelLowerBound}
                            onChange={(e) => setPixelLowerBound(e.target.value)}
                            className="input-field"
                        />
                    </div>
                </>
            )}
            <div className="halfsize-buttons">
                <button className="start-halfsize-btn" onClick={handleStartQuery}>
                    Start Query
                </button>
                <button className="appending-halfsize-btn" onClick={handleShowExecution} disabled={!isQueryActive}>
                    Execution Detail
                </button>
            </div>
            <div className="halfsize-buttons">
                <button className="start-halfsize-btn" onClick={handleStartAugment}>
                    Start Augment
                </button>
                <button className="appending-halfsize-btn" onClick={handleShowPath} disabled={!isPathActive}>
                    Show Path
                </button>
            </div>
            {/* <button onClick={() => {}} disabled={!isQueryActive}>Start Augment</button> */}
            <QueryCommand command={queryCommand} />
        </div>
    );
}

export default InputSection;

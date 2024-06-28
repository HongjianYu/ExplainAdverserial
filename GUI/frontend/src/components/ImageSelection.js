import React, { useState } from 'react';
import QueryCommand from './QueryCommand';
import './InputSection.css';

function InputSection({ scenario, onSearchResults, onModeChange, setExecutionTime, isAug = null, ms = null, setSkippedImages = null, setImagesCount = null }) {
    const [mode, setMode] = useState('Top-K');
    const [k, setK] = useState('5');
    const [pixelUpperBound, setPixelUpperBound] = useState('1');
    const [pixelLowerBound, setPixelLowerBound] = useState('0.5');
    const [order, setOrder] = useState('ASC');
    const [threshold, setThreshold] = useState('0.2');
    const [thresholdDirection, setThresholdDirection] = useState('>');
    const [queryCommand, setQueryCommand] = useState('');
    const [isQueryActive, setIsQueryActive] = useState(false);
    const [isPathActive, setIsPathActive] = useState(false);
    const [latestImageIds, setLatestImageIds] = useState([]);
    const [aug, setAug] = useState(false);

    const roiValue = scenario === 'scenario2' ? 'full image' : 'object bounding box';

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
        handleAugment(false);
        let path, body;
        if (scenario === 'scenario1') {
            if (mode === 'Top-K') {
                path = '/scenario1/api/topk_search';
                body = { k, roi: roiValue, pixelUpperBound, pixelLowerBound, order, ms };
            } else {
                path = '/scenario1/api/filter_search';
                body = { threshold, thresholdDirection, roi: roiValue, pixelUpperBound, pixelLowerBound, ms };
            }
        } else if (scenario === 'scenario2') {
            path = '/scenario2/api/topk_search';
            body = { k, pixelUpperBound, pixelLowerBound, order, ms };
        } else if (scenario === 'scenario3') {
            if (mode === 'Top-K') {
                path = '/scenario3/api/topk_search';
                body = { k, roi: roiValue, pixelUpperBound, pixelLowerBound, order, ms };
            } else if (mode === 'Filter') {
                path = '/scenario3/api/filter_search';
                body = { threshold, thresholdDirection, roi: roiValue, pixelUpperBound, pixelLowerBound, ms };
            }
        }

        const data = await fetchQueryCommand(path, body);
        setQueryCommand(data.query_command);
        onSearchResults(data.image_ids);
        setExecutionTime(data.execution_time);
        if (setSkippedImages) setSkippedImages(data.skipped_images_count);
        if (setImagesCount) setImagesCount(data.images_count);
        setIsQueryActive(true);
        setLatestImageIds(data.image_ids);
    };

    const handleShowPath = () => {
        // Implement later, make a window to show path
    };

    const handleStartAugment = async () => {
        handleAugment(true);
        const response = await fetch('http://localhost:8000/api/augment', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image_ids: latestImageIds }),
        });
        const data = await response.json();
        onSearchResults(data.image_ids);
        setIsPathActive(true);
    };

    const handleModeChange = (newMode) => {
        setMode(newMode);
        setIsQueryActive(false);
        onModeChange(newMode);
    };

    const handleAugment = (newAug) => {
        setAug(newAug);
        if (isAug) isAug(newAug);
    };

    const renderTopKFields = () => (
        <>
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
                <input
                    id="roi"
                    type="text"
                    value={roiValue}
                    className="input-field read-only"
                    readOnly
                />
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
    );

    const renderFilterFields = () => (
        <>
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
                <input
                    id="roi"
                    type="text"
                    value={roiValue}
                    className="input-field read-only"
                    readOnly
                />
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
    );

    const renderScenario1Fields = () => (
        <>
            {mode === 'Top-K' && renderTopKFields()}
            {mode === 'Filter' && renderFilterFields()}
            <div className="halfsize-buttons">
                <button className="start-halfsize-btn" onClick={handleStartQuery}>
                    Start Query
                </button>
                <button className="start-halfsize-btn" onClick={handleStartAugment} disabled={!isQueryActive}>
                    Augment
                </button>
                <button className="appending-halfsize-btn" onClick={handleShowPath} disabled={!isQueryActive}>
                    Show Path
                </button>
            </div>
        </>
    );

    const renderScenario2Fields = () => (
        <>
            {mode === 'Top-K' && renderTopKFields()}
            <div className="halfsize-buttons">
                <button className="start-halfsize-btn" onClick={handleStartQuery}>
                    Start Query
                </button>
            </div>
        </>
    );

    const renderScenario3Fields = () => (
        <>
            {mode === 'Top-K' && renderTopKFields()}
            {mode === 'Filter' && renderFilterFields()}
            <div className="halfsize-buttons">
                <button className="start-halfsize-btn" onClick={handleStartQuery}>
                    Start Query
                </button>
            </div>
        </>
    );

    return (
        <div className="input-section">
            <div className="header">
                <h2 className="title">{mode}</h2>
                <div className="mode-switch">
                    <button className={mode === 'Top-K' ? 'active' : ''} onClick={() => handleModeChange('Top-K')}>Top-K</button>
                    {scenario !== 'scenario2' && (
                        <button className={mode === 'Filter' ? 'active' : ''} onClick={() => handleModeChange('Filter')}>Filter</button>
                    )}
                    {scenario === 'scenario1' && <button className={mode === 'Aggregation' ? 'active' : ''} onClick={() => handleModeChange('Aggregation')}>Aggregation</button>}
                </div>
            </div>
            {scenario === 'scenario1' && renderScenario1Fields()}
            {scenario === 'scenario2' && renderScenario2Fields()}
            {scenario === 'scenario3' && renderScenario3Fields()}
            <QueryCommand command={queryCommand} />
        </div>
    );
}

export default InputSection;

// // src/components/DataPreparation.js
// import React from 'react';
// import { Link } from 'react-router-dom';
// import ConfusionMatrix from './ConfusionMatrix'; // Assuming you export it correctly

// const DataPreparation = () => {
//     return (
        
//         <div>
//             <h1>Data Preparation</h1>
//             <p>Click <Link to="/input">here</Link> to start query.</p>
//             <ConfusionMatrix />
//             {/* Implement image display logic here similar to ResultsSection */}
//         </div>
//     );
// };

import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import CheckPopup from './CheckPopup';
import './DataPreparation.css';

const DataPreparation = () => {
    const [misclassifiedCells, setMisclassifiedCells] = useState([]);
    const [selectedLines, setSelectedLines] = useState({});
    const [isPopupOpen, setIsPopupOpen] = useState(false);

    useEffect(() => {
        // Fetch the pairs from the backend
        const fetchPairs = async () => {
            const response = await fetch('http://localhost:8000/api/get_pairs');
            const data = await response.json();
            setMisclassifiedCells(data);
        };

        fetchPairs();
    }, []);

    const handleToggleSelect = (index) => {
        setSelectedLines(prevSelectedLines => ({
            ...prevSelectedLines,
            [index]: !prevSelectedLines[index]
        }));
    };

    const handleOpenPopup = () => {
        setIsPopupOpen(true);
    };

    const handleClosePopup = () => {
        setIsPopupOpen(false);
    };

    return (
        <div>
            <h1>Top-100 misclassified cells:</h1>
            <div className="misclassified-list">
                {misclassifiedCells.map((cell, index) => (
                    <div key={index} className={`misclassified-line ${selectedLines[index] ? 'selected' : ''}`}>
                        <div className="cell-info">{`${cell.x} predicted as ${cell.y}`}</div>
                        <img src={`http://localhost:8000/topk_results/${cell.x}.jpg`} alt={`Image ${cell.x}`} className="larger-img" />
                        <img src={`http://localhost:8000/topk_results/${cell.x}.jpg`} alt={`Image ${cell.x}`} className="larger-img" />
                        <img src={`http://localhost:8000/topk_results/${cell.y}.jpg`} alt={`Image ${cell.y}`} className="larger-img" />
                        <div className="actions">
                            <button className="custom-btn" onClick={handleOpenPopup}>Check</button>
                            <button className="custom-btn" onClick={() => handleToggleSelect(index)}>
                                {selectedLines[index] ? 'Cancel Select' : 'Select'}
                            </button>
                        </div>
                    </div>
                ))}
            </div>
            <CheckPopup isOpen={isPopupOpen} onClose={handleClosePopup} />
            <p>Click <Link to="/input">here</Link> to start query!</p>
        </div>
    );
};

export default DataPreparation;

import React, { useRef, useEffect } from 'react';
import './OutputPanel.css';

const OutputPanel = ({
    currentMode,
    description,
    detectedObjects,
    rawResponse,
    hoveredObjectId,
    setHoveredObjectId
}) => {
    const objectRefs = useRef({});

    // Scroll to hovered object
    useEffect(() => {
        if (hoveredObjectId && objectRefs.current[hoveredObjectId]) {
            objectRefs.current[hoveredObjectId].scrollIntoView({
                behavior: 'smooth',
                block: 'nearest'
            });
        }
    }, [hoveredObjectId]);

    const renderDescription = () => (
        <div className="output-content">
            <div className="description-container">
                {description ? (
                    <>
                        <h3 className="output-subtitle">Description</h3>
                        <p className="description-text">{description}</p>
                    </>
                ) : (
                    <div className="empty-output">
                        <p className="empty-text">No description generated yet</p>
                        <p className="empty-subtext">Upload an image and click "Process Image"</p>
                    </div>
                )}
            </div>
        </div>
    );

    const renderObjectList = () => (
        <div className="output-content">
            {detectedObjects.length > 0 ? (
                <div className="objects-container">
                    <h3 className="output-subtitle">
                        Detected Objects ({detectedObjects.length})
                    </h3>
                    <div className="objects-list">
                        {detectedObjects.map((obj) => {
                            const isHovered = hoveredObjectId === obj.id;

                            return (
                                <div
                                    key={obj.id}
                                    ref={(el) => (objectRefs.current[obj.id] = el)}
                                    className={`object-item ${isHovered ? 'hovered' : ''}`}
                                    onMouseEnter={() => setHoveredObjectId(obj.id)}
                                    onMouseLeave={() => setHoveredObjectId(null)}
                                >
                                    {/* Class name */}
                                    <div className="object-class">
                                        <span className="class-icon">🎯</span>
                                        <span className="class-name">{obj.label}</span>
                                        <span className="confidence-badge">
                                            {(obj.confidence * 100).toFixed(0)}%
                                        </span>
                                    </div>

                                    {/* Description */}
                                    {obj.description && (
                                        <div className="object-description">
                                            <span className="description-icon">💬</span>
                                            <span className="description-text">{obj.description}</span>
                                        </div>
                                    )}

                                    {/* Bounding box coordinates */}
                                    <div className="object-meta">
                                        <span className="meta-label">BBox:</span>
                                        <span className="meta-value">
                                            [{obj.bbox.map(v => Math.round(v)).join(', ')}]
                                        </span>
                                    </div>

                                    {/* Source */}
                                    {obj.source && (
                                        <div className="object-meta">
                                            <span className="meta-label">Source:</span>
                                            <span className="meta-value source-badge">{obj.source}</span>
                                        </div>
                                    )}
                                </div>
                            );
                        })}
                    </div>
                </div>
            ) : (
                <div className="empty-output">
                    <p className="empty-text">No objects detected yet</p>
                    <p className="empty-subtext">Upload an image and click "Process Image"</p>
                </div>
            )}

            {/* Raw Response (collapsible) */}
            {rawResponse && (
                <details className="raw-response-section">
                    <summary className="raw-response-header">Raw Model Response</summary>
                    <pre className="raw-response-content">{rawResponse}</pre>
                </details>
            )}
        </div>
    );

    return (
        <div className="output-panel">
            <div className="panel-header">
                <h2 className="panel-title">
                    {currentMode === 'description' ? 'Description Output' : 'Detection Results'}
                </h2>
            </div>

            {currentMode === 'description' ? renderDescription() : renderObjectList()}
        </div>
    );
};

export default OutputPanel;

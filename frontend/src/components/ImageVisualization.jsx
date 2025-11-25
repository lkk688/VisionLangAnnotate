import React, { useState, useEffect, useRef } from 'react';
import './ImageVisualization.css';

const ImageVisualization = ({
    imageUrl,
    detectedObjects,
    hoveredObjectId,
    setHoveredObjectId,
    currentMode
}) => {
    const [imageDimensions, setImageDimensions] = useState({ width: 0, height: 0 });
    const [containerDimensions, setContainerDimensions] = useState({ width: 0, height: 0 });
    const imageRef = useRef(null);
    const containerRef = useRef(null);

    // Update container dimensions on resize
    useEffect(() => {
        const updateContainerDimensions = () => {
            if (containerRef.current) {
                const { width, height } = containerRef.current.getBoundingClientRect();
                setContainerDimensions({ width, height });
            }
        };

        updateContainerDimensions();
        window.addEventListener('resize', updateContainerDimensions);

        return () => window.removeEventListener('resize', updateContainerDimensions);
    }, []);

    // Update image dimensions when image loads
    const handleImageLoad = (e) => {
        const { naturalWidth, naturalHeight } = e.target;
        setImageDimensions({ width: naturalWidth, height: naturalHeight });
    };

    // Calculate scale factor for bounding boxes
    const getScaleFactor = () => {
        if (!imageRef.current || !imageDimensions.width) return { scaleX: 1, scaleY: 1, offsetX: 0, offsetY: 0 };

        const { width: displayWidth, height: displayHeight } = imageRef.current.getBoundingClientRect();
        const scaleX = displayWidth / imageDimensions.width;
        const scaleY = displayHeight / imageDimensions.height;

        return { scaleX, scaleY, offsetX: 0, offsetY: 0 };
    };

    // Render bounding boxes
    const renderBoundingBoxes = () => {
        if (!imageRef.current || currentMode !== 'detection' || detectedObjects.length === 0) {
            return null;
        }

        const { scaleX, scaleY } = getScaleFactor();
        const { width: displayWidth, height: displayHeight } = imageRef.current.getBoundingClientRect();

        return (
            <svg
                className="bbox-overlay"
                width={displayWidth}
                height={displayHeight}
                style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    pointerEvents: 'none'
                }}
            >
                {detectedObjects.map((obj) => {
                    const [x1, y1, x2, y2] = obj.bbox;
                    const scaledX = x1 * scaleX;
                    const scaledY = y1 * scaleY;
                    const scaledWidth = (x2 - x1) * scaleX;
                    const scaledHeight = (y2 - y1) * scaleY;
                    const isHovered = hoveredObjectId === obj.id;

                    return (
                        <g key={obj.id}>
                            {/* Bounding box rectangle */}
                            <rect
                                x={scaledX}
                                y={scaledY}
                                width={scaledWidth}
                                height={scaledHeight}
                                className={`bbox ${isHovered ? 'hovered' : ''}`}
                                style={{
                                    pointerEvents: 'auto',
                                    cursor: 'pointer'
                                }}
                                onMouseEnter={() => setHoveredObjectId(obj.id)}
                                onMouseLeave={() => setHoveredObjectId(null)}
                            />

                            {/* Label background */}
                            <rect
                                x={scaledX}
                                y={scaledY - 24}
                                width={Math.min(obj.label.length * 8 + 16, scaledWidth)}
                                height={20}
                                className={`bbox-label-bg ${isHovered ? 'hovered' : ''}`}
                                style={{ pointerEvents: 'none' }}
                            />

                            {/* Label text */}
                            <text
                                x={scaledX + 8}
                                y={scaledY - 10}
                                className="bbox-label-text"
                                style={{ pointerEvents: 'none' }}
                            >
                                {obj.label}
                            </text>

                            {/* Confidence score */}
                            <text
                                x={scaledX + 8}
                                y={scaledY + scaledHeight - 8}
                                className="bbox-confidence-text"
                                style={{ pointerEvents: 'none' }}
                            >
                                {(obj.confidence * 100).toFixed(0)}%
                            </text>
                        </g>
                    );
                })}
            </svg>
        );
    };

    return (
        <div className="image-visualization" ref={containerRef}>
            <div className="visualization-header">
                <h2 className="visualization-title">Image Visualization</h2>
                {detectedObjects.length > 0 && (
                    <span className="object-count">
                        {detectedObjects.length} object{detectedObjects.length !== 1 ? 's' : ''} detected
                    </span>
                )}
            </div>

            <div className="image-container">
                {imageUrl ? (
                    <div className="image-wrapper">
                        <img
                            ref={imageRef}
                            src={imageUrl}
                            alt="Uploaded content"
                            className="main-image"
                            onLoad={handleImageLoad}
                        />
                        {renderBoundingBoxes()}
                    </div>
                ) : (
                    <div className="empty-state">
                        <div className="empty-icon">🖼️</div>
                        <p className="empty-text">No image loaded</p>
                        <p className="empty-subtext">Upload an image to get started</p>
                    </div>
                )}
            </div>
        </div>
    );
};

export default ImageVisualization;

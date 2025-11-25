import React, { useRef } from 'react';
import './ModelControls.css';

const ModelControls = ({
    currentMode,
    setCurrentMode,
    detectionMethod,
    setDetectionMethod,
    useSAM,
    setUseSAM,
    customPrompt,
    setCustomPrompt,
    onImageUpload,
    onProcess,
    loading,
    hasImage
}) => {
    const fileInputRef = useRef(null);

    const handleFileSelect = (e) => {
        const file = e.target.files[0];
        if (file) {
            onImageUpload(file);
        }
    };

    return (
        <div className="model-controls">
            <h2 className="controls-title">Model Setup</h2>

            {/* Function Selector */}
            <div className="control-section">
                <label className="control-label">Function</label>
                <div className="button-group">
                    <button
                        className={`mode-button ${currentMode === 'description' ? 'active' : ''}`}
                        onClick={() => setCurrentMode('description')}
                        disabled={loading}
                    >
                        <span className="button-icon">📝</span>
                        Image Description
                    </button>
                    <button
                        className={`mode-button ${currentMode === 'detection' ? 'active' : ''}`}
                        onClick={() => setCurrentMode('detection')}
                        disabled={loading}
                    >
                        <span className="button-icon">🎯</span>
                        Object Detection
                    </button>
                </div>
            </div>

            {/* Detection Method (only for detection mode) */}
            {currentMode === 'detection' && (
                <div className="control-section">
                    <label className="control-label">Detection Method</label>
                    <select
                        className="control-select"
                        value={detectionMethod}
                        onChange={(e) => setDetectionMethod(e.target.value)}
                        disabled={loading}
                    >
                        <option value="VLM Only">VLM Only</option>
                        <option value="Hybrid Mode">Hybrid Mode</option>
                        <option value="Hybrid-Sequential">Hybrid Sequential</option>
                    </select>
                </div>
            )}

            {/* SAM Segmentation Toggle (only for detection mode) */}
            {currentMode === 'detection' && (
                <div className="control-section">
                    <label className="toggle-container">
                        <input
                            type="checkbox"
                            checked={useSAM}
                            onChange={(e) => setUseSAM(e.target.checked)}
                            disabled={loading}
                        />
                        <span className="toggle-slider"></span>
                        <span className="toggle-label">Enable SAM Segmentation</span>
                    </label>
                </div>
            )}

            {/* Custom Prompt (only for description mode) */}
            {currentMode === 'description' && (
                <div className="control-section">
                    <label className="control-label">Custom Prompt (Optional)</label>
                    <textarea
                        className="control-textarea"
                        placeholder="Enter a custom prompt for image description..."
                        value={customPrompt}
                        onChange={(e) => setCustomPrompt(e.target.value)}
                        disabled={loading}
                        rows={3}
                    />
                </div>
            )}

            {/* Image Upload */}
            <div className="control-section">
                <label className="control-label">Image Upload</label>
                <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    onChange={handleFileSelect}
                    style={{ display: 'none' }}
                    disabled={loading}
                />
                <button
                    className="upload-button"
                    onClick={() => fileInputRef.current?.click()}
                    disabled={loading}
                >
                    <span className="button-icon">📁</span>
                    {hasImage ? 'Change Image' : 'Select Image'}
                </button>
            </div>

            {/* Process Button */}
            <div className="control-section">
                <button
                    className="process-button"
                    onClick={onProcess}
                    disabled={loading || !hasImage}
                >
                    {loading ? (
                        <>
                            <span className="spinner"></span>
                            Processing...
                        </>
                    ) : (
                        <>
                            <span className="button-icon">▶️</span>
                            Process Image
                        </>
                    )}
                </button>
            </div>

            {/* Info Section */}
            <div className="info-section">
                <div className="info-item">
                    <span className="info-label">Status:</span>
                    <span className={`info-value ${hasImage ? 'ready' : 'waiting'}`}>
                        {loading ? 'Processing...' : hasImage ? 'Ready' : 'No image'}
                    </span>
                </div>
            </div>
        </div>
    );
};

export default ModelControls;

import React, { useState } from 'react';
import ModelControls from './ModelControls';
import ImageVisualization from './ImageVisualization';
import OutputPanel from './OutputPanel';
import './VLMInterface.css';

const VLMInterface = () => {
    // State management
    const [currentMode, setCurrentMode] = useState('detection'); // 'description' or 'detection'
    const [selectedImage, setSelectedImage] = useState(null);
    const [imageUrl, setImageUrl] = useState(null);
    const [detectionMethod, setDetectionMethod] = useState('Hybrid Mode');
    const [useSAM, setUseSAM] = useState(true);
    const [customPrompt, setCustomPrompt] = useState('');
    const [loading, setLoading] = useState(false);

    // Results state
    const [description, setDescription] = useState('');
    const [detectedObjects, setDetectedObjects] = useState([]);
    const [rawResponse, setRawResponse] = useState('');

    // Interaction state
    const [hoveredObjectId, setHoveredObjectId] = useState(null);

    // Handle image upload
    const handleImageUpload = async (file) => {
        if (!file) return;

        try {
            setLoading(true);

            // Upload image to backend
            const formData = new FormData();
            formData.append('file', file);

            const uploadResponse = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            const uploadData = await uploadResponse.json();

            if (uploadData.success) {
                setSelectedImage(uploadData.filename);
                setImageUrl(`/api/images/${uploadData.filename}`);
                // Clear previous results
                setDescription('');
                setDetectedObjects([]);
                setRawResponse('');
            } else {
                alert('Error uploading image: ' + uploadData.error);
            }
        } catch (error) {
            console.error('Error uploading image:', error);
            alert('Error uploading image. See console for details.');
        } finally {
            setLoading(false);
        }
    };

    // Process image with selected function
    const handleProcess = async () => {
        if (!selectedImage) {
            alert('Please upload an image first');
            return;
        }

        setLoading(true);

        try {
            if (currentMode === 'description') {
                await processImageDescription();
            } else {
                await processObjectDetection();
            }
        } catch (error) {
            console.error('Error processing image:', error);
            alert('Error processing image. See console for details.');
        } finally {
            setLoading(false);
        }
    };

    // Call image description API
    const processImageDescription = async () => {
        const response = await fetch(`/api/vlm/describe-image/${selectedImage}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                custom_prompt: customPrompt || null
            })
        });

        const data = await response.json();

        if (data.success) {
            setDescription(data.description);
            setDetectedObjects([]);
        } else {
            alert('Error generating description');
        }
    };

    // Call object detection API
    const processObjectDetection = async () => {
        const response = await fetch(`/api/vlm/detect-objects/${selectedImage}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                detection_method: detectionMethod,
                use_sam_segmentation: useSAM
            })
        });

        const data = await response.json();

        if (data.success) {
            // Add unique IDs to objects for hover tracking
            const objectsWithIds = data.objects.map((obj, index) => ({
                ...obj,
                id: `obj-${index}`
            }));
            setDetectedObjects(objectsWithIds);
            setRawResponse(data.raw_response);
            setDescription('');
        } else {
            alert('Error detecting objects');
        }
    };

    return (
        <div className="vlm-interface">
            <div className="vlm-container">
                {/* Left Panel - Model Controls */}
                <ModelControls
                    currentMode={currentMode}
                    setCurrentMode={setCurrentMode}
                    detectionMethod={detectionMethod}
                    setDetectionMethod={setDetectionMethod}
                    useSAM={useSAM}
                    setUseSAM={setUseSAM}
                    customPrompt={customPrompt}
                    setCustomPrompt={setCustomPrompt}
                    onImageUpload={handleImageUpload}
                    onProcess={handleProcess}
                    loading={loading}
                    hasImage={!!selectedImage}
                />

                {/* Center Panel - Image Visualization */}
                <ImageVisualization
                    imageUrl={imageUrl}
                    detectedObjects={detectedObjects}
                    hoveredObjectId={hoveredObjectId}
                    setHoveredObjectId={setHoveredObjectId}
                    currentMode={currentMode}
                />

                {/* Right Panel - Output */}
                <OutputPanel
                    currentMode={currentMode}
                    description={description}
                    detectedObjects={detectedObjects}
                    rawResponse={rawResponse}
                    hoveredObjectId={hoveredObjectId}
                    setHoveredObjectId={setHoveredObjectId}
                />
            </div>
        </div>
    );
};

export default VLMInterface;

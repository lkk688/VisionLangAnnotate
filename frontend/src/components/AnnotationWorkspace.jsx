import React, { useState, useEffect, useRef } from 'react';
import AnnotationCanvas from './AnnotationCanvas';
import AnnotationList from './AnnotationList';
import AnnotationDetails from './AnnotationDetails';
import './AnnotationWorkspace.css';

const AnnotationWorkspace = ({ 
  currentImage, 
  annotations, 
  setAnnotations, 
  onSave,
  loading 
}) => {
  const [selectedAnnotation, setSelectedAnnotation] = useState(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [newCategory, setNewCategory] = useState('person');
  const canvasRef = useRef(null);

  // Reset selected annotation when image changes
  useEffect(() => {
    setSelectedAnnotation(null);
  }, [currentImage]);

  // Handle creating a new bounding box
  const handleCreateBox = () => {
    setIsDrawing(true);
  };

  // Handle annotation selection
  const handleSelectAnnotation = (id) => {
    setSelectedAnnotation(id === selectedAnnotation ? null : id);
  };

  // Handle annotation update
  const handleUpdateAnnotation = (updatedAnnotation) => {
    const updatedAnnotations = annotations.map(ann => 
      ann.id === updatedAnnotation.id ? updatedAnnotation : ann
    );
    setAnnotations(updatedAnnotations);
  };

  // Handle annotation deletion
  const handleDeleteAnnotation = () => {
    if (!selectedAnnotation) return;
    
    const updatedAnnotations = annotations.filter(ann => ann.id !== selectedAnnotation);
    setAnnotations(updatedAnnotations);
    setSelectedAnnotation(null);
  };

  // Handle new annotation creation from canvas
  const handleNewAnnotation = (bbox) => {
    const newId = `ann_${Date.now()}`;
    const newAnnotation = {
      id: newId,
      category: newCategory,
      bbox: bbox,
      score: null
    };
    
    setAnnotations([...annotations, newAnnotation]);
    setIsDrawing(false);
    setSelectedAnnotation(newId);
  };

  return (
    <div className="annotation-workspace">
      {loading && (
        <div className="loading-overlay">
          <div className="spinner-border text-primary" role="status">
            <span className="visually-hidden">Loading...</span>
          </div>
        </div>
      )}
      
      <div className="workspace-header">
        <h4>
          {currentImage ? `Annotating: ${currentImage}` : 'No image selected'}
        </h4>
        <div className="workspace-actions">
          <div className="input-group me-2">
            <label className="input-group-text" htmlFor="categorySelect">Category</label>
            <select 
              className="form-select" 
              id="categorySelect"
              value={newCategory}
              onChange={(e) => setNewCategory(e.target.value)}
              disabled={!currentImage}
            >
              <option value="person">Person</option>
              <option value="car">Car</option>
              <option value="bicycle">Bicycle</option>
              <option value="motorcycle">Motorcycle</option>
              <option value="truck">Truck</option>
              <option value="traffic light">Traffic Light</option>
              <option value="other">Other</option>
            </select>
          </div>
          
          <button 
            className="btn btn-primary me-2" 
            onClick={handleCreateBox}
            disabled={!currentImage || isDrawing}
          >
            Create Box
          </button>
          
          <button 
            className="btn btn-danger me-2" 
            onClick={handleDeleteAnnotation}
            disabled={!selectedAnnotation}
          >
            Delete Annotation
          </button>
          
          <button 
            className="btn btn-success" 
            onClick={onSave}
            disabled={!currentImage}
          >
            Save
          </button>
        </div>
      </div>
      
      <div className="workspace-content">
        <div className="canvas-container">
          {currentImage ? (
            <AnnotationCanvas
              ref={canvasRef}
              imageSrc={`/api/images/${currentImage}`}
              annotations={annotations}
              selectedAnnotation={selectedAnnotation}
              isDrawing={isDrawing}
              onNewAnnotation={handleNewAnnotation}
              onSelectAnnotation={handleSelectAnnotation}
            />
          ) : (
            <div className="no-image-placeholder">
              <p>Select an image from the sidebar to start annotating</p>
            </div>
          )}
        </div>
        
        <div className="annotation-panel">
          <div className="annotation-list-container">
            <h5>Annotations</h5>
            <AnnotationList
              annotations={annotations}
              selectedAnnotation={selectedAnnotation}
              onSelectAnnotation={handleSelectAnnotation}
            />
          </div>
          
          <div className="annotation-details-container">
            <h5>Details</h5>
            <AnnotationDetails
              annotation={annotations.find(ann => ann.id === selectedAnnotation)}
              onUpdate={handleUpdateAnnotation}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default AnnotationWorkspace;
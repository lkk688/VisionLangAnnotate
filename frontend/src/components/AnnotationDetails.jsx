import React, { useState, useEffect } from 'react';
import './AnnotationDetails.css';

const AnnotationDetails = ({ annotation, onUpdate }) => {
  const [formData, setFormData] = useState({
    category: '',
    x: 0,
    y: 0,
    width: 0,
    height: 0
  });

  // Update form data when annotation changes
  useEffect(() => {
    if (annotation) {
      const [x, y, width, height] = annotation.bbox;
      setFormData({
        category: annotation.category,
        x: Math.round(x),
        y: Math.round(y),
        width: Math.round(width),
        height: Math.round(height)
      });
    }
  }, [annotation]);

  // Handle form input changes
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: name === 'category' ? value : parseInt(value, 10)
    }));
  };

  // Handle form submission
  const handleSubmit = (e) => {
    e.preventDefault();
    
    if (!annotation) return;
    
    const updatedAnnotation = {
      ...annotation,
      category: formData.category,
      bbox: [formData.x, formData.y, formData.width, formData.height]
    };
    
    onUpdate(updatedAnnotation);
  };

  if (!annotation) {
    return (
      <div className="annotation-details">
        <p className="text-muted">No annotation selected</p>
      </div>
    );
  }

  return (
    <div className="annotation-details">
      <form onSubmit={handleSubmit}>
        <div className="mb-3">
          <label htmlFor="category" className="form-label">Category</label>
          <input
            type="text"
            className="form-control"
            id="category"
            name="category"
            value={formData.category}
            onChange={handleChange}
          />
        </div>
        
        <div className="row">
          <div className="col-6 mb-3">
            <label htmlFor="x" className="form-label">X</label>
            <input
              type="number"
              className="form-control"
              id="x"
              name="x"
              value={formData.x}
              onChange={handleChange}
            />
          </div>
          
          <div className="col-6 mb-3">
            <label htmlFor="y" className="form-label">Y</label>
            <input
              type="number"
              className="form-control"
              id="y"
              name="y"
              value={formData.y}
              onChange={handleChange}
            />
          </div>
        </div>
        
        <div className="row">
          <div className="col-6 mb-3">
            <label htmlFor="width" className="form-label">Width</label>
            <input
              type="number"
              className="form-control"
              id="width"
              name="width"
              value={formData.width}
              onChange={handleChange}
            />
          </div>
          
          <div className="col-6 mb-3">
            <label htmlFor="height" className="form-label">Height</label>
            <input
              type="number"
              className="form-control"
              id="height"
              name="height"
              value={formData.height}
              onChange={handleChange}
            />
          </div>
        </div>
        
        <button type="submit" className="btn btn-primary">Update</button>
      </form>
    </div>
  );
};

export default AnnotationDetails;
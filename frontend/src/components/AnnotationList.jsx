import React from 'react';
import './AnnotationList.css';

const AnnotationList = ({ annotations, selectedAnnotation, onSelectAnnotation }) => {
  // Utility function to generate a color for a category
  const getColorForCategory = (category) => {
    // Simple hash function to generate a consistent color for a given category
    let hash = 0;
    for (let i = 0; i < category.length; i++) {
      hash = category.charCodeAt(i) + ((hash << 5) - hash);
    }
    
    // Convert to RGB
    const r = (hash & 0xFF0000) >> 16;
    const g = (hash & 0x00FF00) >> 8;
    const b = hash & 0x0000FF;
    
    return `rgb(${r}, ${g}, ${b})`;
  };

  return (
    <div className="annotation-list">
      {annotations.length === 0 ? (
        <p className="text-muted">No annotations for this image</p>
      ) : (
        <div className="list-group">
          {annotations.map(ann => (
            <a
              key={ann.id}
              href="#"
              className={`list-group-item list-group-item-action d-flex justify-content-between align-items-center ${selectedAnnotation === ann.id ? 'active' : ''}`}
              onClick={(e) => {
                e.preventDefault();
                onSelectAnnotation(ann.id);
              }}
            >
              <div className="d-flex align-items-center">
                <span 
                  className="category-badge"
                  style={{ backgroundColor: getColorForCategory(ann.category) }}
                >
                  {ann.category}
                </span>
                
                {ann.score && (
                  <small className="text-muted ms-2">
                    ({ann.score.toFixed(2)})
                  </small>
                )}
              </div>
            </a>
          ))}
        </div>
      )}
    </div>
  );
};

export default AnnotationList;
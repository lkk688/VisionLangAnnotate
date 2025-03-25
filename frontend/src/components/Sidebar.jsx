import React, { useRef } from 'react';
import './Sidebar.css';

const Sidebar = ({ images, currentImage, onImageSelect, onRefreshImages }) => {
  const fileInputRef = useRef(null);
  const annotationFileInputRef = useRef(null);

  // Handle image upload
  const handleImageUpload = async (event) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append('files', files[i]);
    }

    try {
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();
      
      if (data.success) {
        alert(`Uploaded ${data.uploaded} of ${files.length} images.`);
        onRefreshImages();
      } else {
        alert('Error uploading images: ' + data.error);
      }
    } catch (error) {
      console.error('Error uploading images:', error);
      alert('Error uploading images. See console for details.');
    }
    
    // Reset file input
    event.target.value = null;
  };

  // Handle annotation file upload
  const handleAnnotationUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/upload-annotations', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();
      
      if (data.success) {
        alert('Annotations uploaded successfully!');
        
        // Reload current image if any
        if (currentImage) {
          onImageSelect(currentImage);
        }
      } else {
        alert('Error uploading annotations: ' + data.error);
      }
    } catch (error) {
      console.error('Error uploading annotations:', error);
      alert('Error uploading annotations. See console for details.');
    }
    
    // Reset file input
    event.target.value = null;
  };

  // Handle image deletion
  const handleDeleteImage = async (imageName) => {
    if (!imageName) return;
    
    if (!window.confirm(`Are you sure you want to delete ${imageName}?`)) {
      return;
    }

    try {
      const response = await fetch(`/api/delete-image/${imageName}`, {
        method: 'DELETE'
      });

      const data = await response.json();
      
      if (data.success) {
        alert(`Image ${imageName} deleted successfully.`);
        onRefreshImages();
      } else {
        alert('Error deleting image: ' + data.error);
      }
    } catch (error) {
      console.error('Error deleting image:', error);
      alert('Error deleting image. See console for details.');
    }
  };

  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <h5>Images</h5>
        <div className="sidebar-actions">
          <button 
            className="btn btn-sm btn-primary" 
            onClick={() => fileInputRef.current.click()}
          >
            Upload Images
          </button>
          <input
            type="file"
            ref={fileInputRef}
            style={{ display: 'none' }}
            onChange={handleImageUpload}
            multiple
            accept="image/*"
          />
          
          <button 
            className="btn btn-sm btn-secondary mt-2" 
            onClick={() => annotationFileInputRef.current.click()}
          >
            Import Annotations
          </button>
          <input
            type="file"
            ref={annotationFileInputRef}
            style={{ display: 'none' }}
            onChange={handleAnnotationUpload}
            accept=".json"
          />
        </div>
      </div>
      
      <div className="image-list">
        {images.length === 0 ? (
          <p className="text-muted">No images uploaded</p>
        ) : (
          <div className="list-group">
            {images.map((image) => (
              <div 
                key={image} 
                className={`list-group-item list-group-item-action d-flex justify-content-between align-items-center ${currentImage === image ? 'active' : ''}`}
                onClick={() => onImageSelect(image)}
              >
                <span className="image-name">{image}</span>
                <button 
                  className="btn btn-sm btn-danger"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDeleteImage(image);
                  }}
                >
                  Delete
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default Sidebar;
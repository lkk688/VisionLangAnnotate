// frontend/src/components/ImageDisplay.jsx
function ImageDisplay({ imageUrl }) {
    return (
      <div className="image-container">
        <img 
          src={imageUrl} 
          alt="Uploaded content" 
          style={{ maxWidth: '100%' }}
        />
      </div>
    );
  }
  
  export default ImageDisplay;
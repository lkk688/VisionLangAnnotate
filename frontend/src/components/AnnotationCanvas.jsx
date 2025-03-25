import React, { useRef, useEffect, useState, forwardRef, useImperativeHandle } from 'react';
import './AnnotationCanvas.css';

const AnnotationCanvas = forwardRef(({
  imageSrc,
  annotations,
  selectedAnnotation,
  isDrawing,
  onNewAnnotation,
  onSelectAnnotation
}, ref) => {
  const canvasRef = useRef(null);
  const imageRef = useRef(null);
  const [image, setImage] = useState(null);
  const [canvasSize, setCanvasSize] = useState({ width: 0, height: 0 });
  const [drawStart, setDrawStart] = useState(null);
  const [drawCurrent, setDrawCurrent] = useState(null);
  
  // Expose canvas methods to parent component
  useImperativeHandle(ref, () => ({
    getCanvas: () => canvasRef.current,
    getContext: () => canvasRef.current.getContext('2d'),
    getCanvasSize: () => canvasSize
  }));

  // Load image when source changes
  useEffect(() => {
    if (!imageSrc) return;
    
    const img = new Image();
    img.src = imageSrc;
    img.onload = () => {
      setImage(img);
      
      // Calculate canvas size to fit the container while maintaining aspect ratio
      const container = canvasRef.current.parentElement;
      const containerWidth = container.clientWidth;
      const containerHeight = container.clientHeight;
      
      const imgAspectRatio = img.width / img.height;
      const containerAspectRatio = containerWidth / containerHeight;
      
      let canvasWidth, canvasHeight;
      
      if (imgAspectRatio > containerAspectRatio) {
        // Image is wider than container
        canvasWidth = containerWidth;
        canvasHeight = containerWidth / imgAspectRatio;
      } else {
        // Image is taller than container
        canvasHeight = containerHeight;
        canvasWidth = containerHeight * imgAspectRatio;
      }
      
      setCanvasSize({ width: canvasWidth, height: canvasHeight });
    };
    
    img.onerror = () => {
      console.error('Error loading image');
    };
    
    return () => {
      img.onload = null;
      img.onerror = null;
    };
  }, [imageSrc]);

  // Draw annotations when they change
  useEffect(() => {
    if (!canvasRef.current || !image) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw image
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    
    // Draw annotations
    annotations.forEach(ann => {
      const [x, y, width, height] = ann.bbox;
      
      // Scale coordinates to canvas size
      const scaledX = (x / image.width) * canvas.width;
      const scaledY = (y / image.height) * canvas.height;
      const scaledWidth = (width / image.width) * canvas.width;
      const scaledHeight = (height / image.height) * canvas.height;
      
      // Set style based on selection state
      ctx.lineWidth = ann.id === selectedAnnotation ? 3 : 2;
      ctx.strokeStyle = getColorForCategory(ann.category);
      
      // Draw rectangle
      ctx.strokeRect(scaledX, scaledY, scaledWidth, scaledHeight);
      
      // Draw label if it's selected or has a score
      if (ann.id === selectedAnnotation || ann.score) {
        ctx.font = '12px Arial';
        ctx.fillStyle = getColorForCategory(ann.category);
        
        let labelText = ann.category;
        if (ann.score) {
          labelText += ` (${ann.score.toFixed(2)})`;
        }
        
        ctx.fillText(labelText, scaledX, scaledY - 5);
      }
    });
    
    // Draw current rectangle if drawing
    if (isDrawing && drawStart && drawCurrent) {
      ctx.lineWidth = 2;
      ctx.strokeStyle = 'red';
      
      const width = drawCurrent.x - drawStart.x;
      const height = drawCurrent.y - drawStart.y;
      
      ctx.strokeRect(drawStart.x, drawStart.y, width, height);
    }
  }, [annotations, selectedAnnotation, image, canvasSize, isDrawing, drawStart, drawCurrent]);

  // Handle mouse events for drawing
  const handleMouseDown = (e) => {
    if (!isDrawing || !image) return;
    
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    setDrawStart({ x, y });
    setDrawCurrent({ x, y });
  };

  const handleMouseMove = (e) => {
    if (!isDrawing || !drawStart || !image) return;
    
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    setDrawCurrent({ x, y });
  };

  const handleMouseUp = (e) => {
    if (!isDrawing || !drawStart || !drawCurrent || !image) return;
    
    // Calculate bounding box in image coordinates
    const x = Math.min(drawStart.x, drawCurrent.x) / canvasSize.width * image.width;
    const y = Math.min(drawStart.y, drawCurrent.y) / canvasSize.height * image.height;
    const width = Math.abs(drawCurrent.x - drawStart.x) / canvasSize.width * image.width;
    const height = Math.abs(drawCurrent.y - drawStart.y) / canvasSize.height * image.height;
    
    // Create new annotation if box is large enough
    if (width > 5 && height > 5) {
      onNewAnnotation([x, y, width, height]);
    }
    
    // Reset drawing state
    setDrawStart(null);
    setDrawCurrent(null);
  };

  // Handle click to select annotation
  const handleClick = (e) => {
    if (isDrawing || !image) return;
    
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    // Convert to image coordinates
    const imageX = x / canvasSize.width * image.width;
    const imageY = y / canvasSize.height * image.height;
    
    // Find clicked annotation (in reverse order to select top-most first)
    for (let i = annotations.length - 1; i >= 0; i--) {
      const ann = annotations[i];
      const [annX, annY, annWidth, annHeight] = ann.bbox;
      
      if (
        imageX >= annX && 
        imageX <= annX + annWidth && 
        imageY >= annY && 
        imageY <= annY + annHeight
      ) {
        onSelectAnnotation(ann.id);
        return;
      }
    }
    
    // If no annotation was clicked, deselect
    onSelectAnnotation(null);
  };

  return (
    <canvas
      ref={canvasRef}
      width={canvasSize.width}
      height={canvasSize.height}
      className="annotation-canvas"
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onClick={handleClick}
    />
  );
});

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

export default AnnotationCanvas;
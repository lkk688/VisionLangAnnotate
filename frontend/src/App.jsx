import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Sidebar from './components/Sidebar';
import AnnotationWorkspace from './components/AnnotationWorkspace';
import './App.css';

function App() {
  const [images, setImages] = useState([]);
  const [currentImage, setCurrentImage] = useState(null);
  const [annotations, setAnnotations] = useState([]);
  const [loading, setLoading] = useState(false);

  // Fetch images on component mount
  useEffect(() => {
    fetchImages();
  }, []);

  // Fetch images from the server
  const fetchImages = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/images');
      const data = await response.json();
      
      if (data.success) {
        setImages(data.images);
      } else {
        console.error('Error fetching images:', data.error);
      }
    } catch (error) {
      console.error('Error fetching images:', error);
    } finally {
      setLoading(false);
    }
  };

  // Load image and its annotations
  const loadImage = async (imageName) => {
    if (!imageName) return;
    
    try {
      setLoading(true);
      const response = await fetch(`/api/annotations/${imageName}`);
      const data = await response.json();
      
      if (data.success) {
        setCurrentImage(imageName);
        setAnnotations(data.annotations || []);
      } else {
        console.error('Error loading annotations:', data.error);
      }
    } catch (error) {
      console.error('Error loading annotations:', error);
    } finally {
      setLoading(false);
    }
  };

  // Save annotations for the current image
  const saveAnnotations = async () => {
    if (!currentImage) return;
    
    try {
      setLoading(true);
      const response = await fetch('/api/save-annotation', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          image: currentImage,
          annotations: annotations
        })
      });
      
      const data = await response.json();
      
      if (data.success) {
        alert('Annotations saved successfully!');
      } else {
        alert('Error saving annotations: ' + data.error);
      }
    } catch (error) {
      console.error('Error saving annotations:', error);
      alert('Error saving annotations. See console for details.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Router>
      <div className="app">
        <Navbar />
        <div className="main-container">
          <Sidebar 
            images={images} 
            currentImage={currentImage}
            onImageSelect={loadImage}
            onRefreshImages={fetchImages}
          />
          <Routes>
            <Route 
              path="/" 
              element={
                <AnnotationWorkspace 
                  currentImage={currentImage}
                  annotations={annotations}
                  setAnnotations={setAnnotations}
                  onSave={saveAnnotations}
                  loading={loading}
                />
              } 
            />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;

// Image Display Example
// import { useState, useEffect } from 'react';
// import ImageDisplay from './components/ImageDisplay';

// function App() {
//   // Fetch this from your backend API (/api/images) or state
//   const imageUrl = "http://localhost:8000/static/uploads/test.jpg";

//   return (
//     <div>
//       <h1>My Annotation Tool</h1>
//       <ImageDisplay imageUrl={imageUrl} />
//     </div>
//   );
// }

// function App() {
//   const [message, setMessage] = useState('');

//   useEffect(() => {
//     fetch('http://localhost:8000')
//       .then((res) => res.json())
//       .then((data) => setMessage(data.message));
//   }, []);

//   return <h1>{message || "Loading..."}</h1>;
// }

//export default App;

// import { useState } from 'react'
// import reactLogo from './assets/react.svg'
// import viteLogo from '/vite.svg'
// import './App.css'

// function App() {
//   const [count, setCount] = useState(0)

//   return (
//     <>
//       <div>
//         <a href="https://vite.dev" target="_blank">
//           <img src={viteLogo} className="logo" alt="Vite logo" />
//         </a>
//         <a href="https://react.dev" target="_blank">
//           <img src={reactLogo} className="logo react" alt="React logo" />
//         </a>
//       </div>
//       <h1>Vite + React</h1>
//       <div className="card">
//         <button onClick={() => setCount((count) => count + 1)}>
//           count is {count}
//         </button>
//         <p>
//           Edit <code>src/App.jsx</code> and save to test HMR
//         </p>
//       </div>
//       <p className="read-the-docs">
//         Click on the Vite and React logos to learn more
//       </p>
//     </>
//   )
// }

// export default App

import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import VLMInterface from './components/VLMInterface';
import './App.css';

function App() {
  return (
    <Router>
      <div className="app">
        <Routes>
          <Route path="/" element={<VLMInterface />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;

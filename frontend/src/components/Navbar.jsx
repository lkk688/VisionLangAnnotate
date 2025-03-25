import React from 'react';
import { Link } from 'react-router-dom';
import './Navbar.css';

const Navbar = () => {
  return (
    <nav className="navbar navbar-expand-lg navbar-dark bg-dark">
      <div className="container-fluid">
        <Link className="navbar-brand" to="/">
          Simple Label Studio
        </Link>
        <button
          className="navbar-toggler"
          type="button"
          data-bs-toggle="collapse"
          data-bs-target="#navbarNav"
          aria-controls="navbarNav"
          aria-expanded="false"
          aria-label="Toggle navigation"
        >
          <span className="navbar-toggler-icon"></span>
        </button>
        <div className="collapse navbar-collapse" id="navbarNav">
          <ul className="navbar-nav me-auto">
            <li className="nav-item">
              <Link className="nav-link" to="/">
                Annotation
              </Link>
            </li>
          </ul>
          <div className="d-flex">
            <div className="dropdown">
              <button
                className="btn btn-outline-light dropdown-toggle"
                type="button"
                id="exportDropdown"
                data-bs-toggle="dropdown"
                aria-expanded="false"
              >
                Export
              </button>
              <ul className="dropdown-menu dropdown-menu-end" aria-labelledby="exportDropdown">
                <li>
                  <button className="dropdown-item" onClick={() => exportAnnotations('json')}>
                    Export as JSON
                  </button>
                </li>
                <li>
                  <button className="dropdown-item" onClick={() => exportAnnotations('coco')}>
                    Export as COCO
                  </button>
                </li>
                <li>
                  <button className="dropdown-item" onClick={() => exportAnnotations('yolo')}>
                    Export as YOLO
                  </button>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
};

// Function to export annotations
const exportAnnotations = async (format) => {
  try {
    const response = await fetch('/api/export', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ format })
    });
    
    const data = await response.json();
    
    if (data.success) {
      alert(`Annotations exported successfully in ${data.format} format!\nSaved to: ${data.path}`);
    } else {
      alert('Error exporting annotations: ' + data.error);
    }
  } catch (error) {
    console.error('Error exporting annotations:', error);
    alert('Error exporting annotations. See console for details.');
  }
};

export default Navbar;
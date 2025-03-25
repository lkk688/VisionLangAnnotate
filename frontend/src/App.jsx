import { useState, useEffect } from 'react';
import ImageDisplay from './components/ImageDisplay';

function App() {
  // Fetch this from your backend API (/api/images) or state
  const imageUrl = "http://localhost:8000/static/uploads/test.jpg";

  return (
    <div>
      <h1>My Annotation Tool</h1>
      <ImageDisplay imageUrl={imageUrl} />
    </div>
  );
}
// function App() {
//   const [message, setMessage] = useState('');

//   useEffect(() => {
//     fetch('http://localhost:8000')
//       .then((res) => res.json())
//       .then((data) => setMessage(data.message));
//   }, []);

//   return <h1>{message || "Loading..."}</h1>;
// }

export default App;

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

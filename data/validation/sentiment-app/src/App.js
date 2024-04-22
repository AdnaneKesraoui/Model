import React from 'react';
import './App.css';
import SentimentPredictor from './SentimentPredictor';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Sentiment Analysis App</h1>
      </header>
      <SentimentPredictor />
    </div>
  );
}

export default App;

import axios from 'axios';
import React, { useState } from 'react';

function SentimentPredictor() {
  const [text, setText] = useState('');
  const [sentiment, setSentiment] = useState(null);

  const handleInputChange = (event) => {
    setText(event.target.value);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    try {
      const response = await axios.post(' https://employee-reviews-model-yyxax4dhpq-no.a.run.app/predict', { text });
      setSentiment(response.data.sentiment);
    } catch (error) {
      console.error('Error fetching sentiment:', error);
      setSentiment('Error fetching sentiment');
    }
  };

  return (
    <div>
      <h1>Sentiment Predictor</h1>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={text}
          onChange={handleInputChange}
          placeholder="Enter text here..."
        />
        <button type="submit">Predict Sentiment</button>
      </form>
      {sentiment !== null && <h2>Sentiment: {sentiment}</h2>}
    </div>
  );
}

export default SentimentPredictor;

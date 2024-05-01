import { Box, Button, TextField, Typography } from '@material-ui/core';
import axios from 'axios';
import React, { useState } from 'react';

function SentimentPredictor() {
  const [text, setText] = useState('');
  const [sentiment, setSentiment] = useState(null);
  const [error, setError] = useState(null);

  const handleInputChange = (event) => {
    setText(event.target.value);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    try {
      const response = await axios.post('https://employee-reviews-updated-yyxax4dhpq-no.a.run.app/predict', { text });
      setSentiment(response.data.sentiment);
      setError(null);
    } catch (error) {
      console.error('Error fetching sentiment:', error);
      setSentiment(null);
      setError('Error fetching sentiment');
    }
  };

  return (
    <Box display="flex" flexDirection="column" alignItems="center" mt={4}>
      <Typography variant="h4" gutterBottom>Employee Reviews</Typography>
      <form onSubmit={handleSubmit}>
        <TextField
          variant="outlined"
          margin="normal"
          required
          fullWidth
          id="text"
          label="Enter text here..."
          name="text"
          value={text}
          onChange={handleInputChange}
        />
        <Button
          type="submit"
          variant="contained"
          color="primary"
          fullWidth
          style={{ marginTop: '1rem' }}
        >
          Submit
        </Button>
      </form>
      {sentiment && (
        <Typography variant="h5" style={{ marginTop: '1rem' }}>
          Sentiment: {sentiment}
        </Typography>
      )}
      {error && (
        <Typography variant="h5" color="error" style={{ marginTop: '1rem' }}>
          {error}
        </Typography>
      )}
    </Box>
  );
}

export default SentimentPredictor;

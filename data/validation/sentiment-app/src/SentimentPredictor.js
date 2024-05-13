import { Box, Button, Input, Typography } from '@material-ui/core';
import axios from 'axios';
import React, { useState } from 'react';

function SentimentPredictor() {
  const [file, setFile] = useState(null);
  const [results, setResults] = useState([]);
  const [error, setError] = useState(null);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!file) return;
    setError(null);
    setResults([]);
    try {
      const reader = new FileReader();
      reader.onload = async (e) => {
        const text = e.target.result;
        const lines = text.split('\n');
        let allResults = [];
        for (const line of lines) {
          if (line.trim()) {  // Adding trim to skip empty lines
            try {
              const response = await axios.post('http://127.0.0.1:8000/predict', { text: line });
              if (Array.isArray(response.data)) {
                allResults = allResults.concat(response.data);  // Concatenating all results into a single array
              } else {
                console.error('Received data is not an array:', response.data);
              }
            } catch (error) {
              console.error('Error fetching sentiment:', error);
              setError('Error fetching sentiment');
              break;
            }
          }
        }
        setResults(allResults);  // Updating the state only once after all lines are processed
      };
      reader.readAsText(file);
    } catch (error) {
      console.error('Error handling the file:', error);
      setError('Error processing the file');
    }
  };

  return (
    <Box display="flex" flexDirection="column" alignItems="center" mt={4}>
      <Typography variant="h4" gutterBottom>Employee Reviews</Typography>
      <form onSubmit={handleSubmit}>
        <Input
          type="file"
          inputProps={{ accept: '.txt' }}
          onChange={handleFileChange}
          required
        />
        <Button
          type="submit"
          variant="contained"
          color="primary"
          style={{ marginTop: '1rem' }}
        >
          Analyze Sentiments
        </Button>
      </form>
      {results.length > 0 && (
        results.map((result, index) => (
          <Typography key={index} variant="h5" style={{ marginTop: '1rem' }}>
            Text: {result.text} - Sentiment: {result.sentiment}
          </Typography>
        ))
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

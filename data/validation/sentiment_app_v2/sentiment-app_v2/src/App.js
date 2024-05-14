import { LoadingButton } from '@mui/lab';
import { Button, CircularProgress, Paper, Typography } from '@mui/material'; // Import Button component
import axios from 'axios';
import React, { useState } from 'react';

function App() {
  const [file, setFile] = useState(null);
  const [results, setResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [sentimentCounts, setSentimentCounts] = useState({ positive: 0, negative: 0, neutral: 0 });

  const onFileChange = event => {
    setFile(event.target.files[0]);
  };

  const onFileUpload = async () => {
    setIsLoading(true);
    const formData = new FormData();
    formData.append("file", file);
    
    try {
      const response = await axios.post("https://employee-reviews-latest-yyxax4dhpq-no.a.run.app/predict", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      setResults(response.data);
      calculateSentimentCounts(response.data);
    } catch (error) {
      console.error('Error uploading file:', error);
    }
    setIsLoading(false);
  };

  const calculateSentimentCounts = (data) => {
    const counts = data.reduce((acc, curr) => {
      acc[curr.sentiment] = (acc[curr.sentiment] || 0) + 1;
      return acc;
    }, { positive: 0, negative: 0, neutral: 0 });

    setSentimentCounts({
      positive: (counts.positive / data.length) * 100,
      negative: (counts.negative / data.length) * 100,
      neutral: (counts.neutral / data.length) * 100,
    });
  };

  const fileData = () => {
    if (file) {
      return (
        <Paper style={{ padding: '20px', marginTop: '20px' }}>
          <Typography variant="h6">File Details:</Typography>
          <Typography>Name: {file.name}</Typography>
          <Typography>Type: {file.type}</Typography>
          <Typography>Last Modified: {file.lastModifiedDate.toDateString()}</Typography>
        </Paper>
      );
    }
    return null;
  };

  return (
    <div style={{ padding: '40px' }}>
      <Typography variant="h4" gutterBottom>
        Sentiment Analysis
      </Typography>
      <Typography variant="subtitle1" gutterBottom>
        Upload your Text File
      </Typography>
      <input type="file" hidden id="contained-button-file" onChange={onFileChange} />
      <label htmlFor="contained-button-file">
        <Button variant="contained" component="span" style={{ marginRight: '10px' }}>
          Upload File
        </Button>
      </label>
      <LoadingButton
        onClick={onFileUpload}
        loading={isLoading}
        loadingIndicator={<CircularProgress size={24} />}
        variant="contained"
        color="primary"
      >
        Analyze
      </LoadingButton>
      {fileData()}
      {results.length > 0 && (
        <Paper style={{ marginTop: '20px', padding: '20px' }}>
          <Typography variant="h6">Sentiment Analysis Results:</Typography>
          <Typography>Positive: {sentimentCounts.positive.toFixed(2)}%</Typography>
          <Typography>Negative: {sentimentCounts.negative.toFixed(2)}%</Typography>
          <Typography>Neutral: {sentimentCounts.neutral.toFixed(2)}%</Typography>
        </Paper>
      )}
    </div>
  );
}

export default App;
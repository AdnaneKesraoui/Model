import { AppBar, Box, Container, CssBaseline, Toolbar, Typography } from '@material-ui/core';
import React from 'react';
import SentimentPredictor from './SentimentPredictor';

function App() {
  return (
    <div>
      <CssBaseline />
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6">Sentiment Analysis App</Typography>
        </Toolbar>
      </AppBar>
      <Container component="main" maxWidth="lg">
        <Box my={4}>
          <SentimentPredictor />
        </Box>
      </Container>
    </div>
  );
}

export default App;

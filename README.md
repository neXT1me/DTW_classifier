# DTW Classifier
Implementation of the classifier based on the DTW algorithm

## Algorithm Description
The DTW (Dynamic Time Warping) classifier is designed to compare and align two time series sequences by calculating an optimal match between them, even if they differ in speed or timing. The algorithm uses dynamic programming to find the minimum distance path, enabling effective classification of time series data.

## Formulas Used
The classification process involves the following key formulas:

1. **Threshold Calculation**:
   $ d_{threshold} = \frac{\sum_{i=1}^{n} d(g_i, g_{ref})}{n} + 3 \cdot \sigma $
   where $ d(g_i, g_{ref}) $ is the DTW distance between the input sequence $ g_i $ and the reference sequence $ g_{ref} $, and $ n $ is the number of sequences.

2. **Standard Deviation ($\sigma$)**:
   $ \sigma = \sqrt{\frac{\sum_{i=1}^{n} [d(g_i, g_{ref}) - \frac{\sum_{j=1}^{n} d(g_j, g_{ref})}{n}]^2}{n}} $
   This measures the variability of the DTW distances from the mean.

The algorithm classifies a sequence based on whether its DTW distance to the reference falls below the calculated threshold, indicating a match.

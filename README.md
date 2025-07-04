# DTW Classifier
Implementation of the classifier based on the DTW algorithm

## Algorithm Description
The DTW (Dynamic Time Warping) classifier is crafted to align and compare two time series by determining the best match, accommodating variations in pace or timing. It employs dynamic programming to identify the shortest distance path, facilitating efficient time series classification.

## Formulas Used
The classification process involves the following key formulas:

1. **Threshold Calculation**:
   $$ d_{threshold} = \frac{\sum_{i=1}^{n} d(g_i, g_{ref})}{n} + 3 \cdot \sigma $$
   where \( d(g_i, g_{ref}) \) represents the DTW distance between the input sequence \( g_i \) and the reference sequence \( g_{ref} \), and \( n \) denotes the total number of sequences.

2. **Standard Deviation (\(\sigma\))**:
   $$ \sigma = \sqrt{\frac{\sum_{i=1}^{n} [d(g_i, g_{ref}) - \frac{\sum_{j=1}^{n} d(g_j, g_{ref})}{n}]^2}{n}} $$
   This quantifies the dispersion of DTW distances around the average.

The algorithm determines a sequence's classification by checking if its DTW distance to the reference stays under the computed threshold, signifying a match.

**The Cauchy-Schwarz Inequality**\
$$\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)$$

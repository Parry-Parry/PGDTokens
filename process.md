
# PGD 

Performing projected gradient descent over a bi-encoder BERT model, aim is to find what token placed at the start, maximises relevance across all documents and queries 

## Inputs
* query 
* documents

## Process
* add a placeholder token to the start of every document
* Perform projected gradient descent to optimise for each document with placeholder token to be the most relevant document
* Eventually get final embedding

## Outputs
* An optimal embedding over all qd pairs
* The candidate tokens which will most likely improve grad
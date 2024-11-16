### comp4222-project20 Edge Prediction of Trust Relationship on Epinions

## Group members

darkmmon: YIU, Tin Cheung Ivan (Ivan)
lkmremy: LEUNG, Kam Ming (Jeremy)
DavidSitu: SITU, Kam Chung (David)

## Dataset

Data is collected as a subset of [this dataset](https://snap.stanford.edu/data/soc-sign-epinions.html).
This is a who-trust-whom online social network of a a general consumer review site Epinions.com.

## Model

We would use Graph Attention Network (GAT) to carry out link prediction.
We first processed the data into graphs and initialized with learnable embeddings, then used the GATv2Conv model from PyTorch.

## Usage

1. Un-zip the file in `/data`
2. Run the code in `main.ipynb`

## File Structure

`/data/soc-sign-epinions.txt.gz` and `/data/soc-sign-epinions.txt` - dataset
`main.ipynb` - main code for loading data, encoding and decoding, GAT model and validation
`EarlyStopping.py` - implements a threshold to stop training at low improvement
`checkpoint.pt`(generated) - saves the state of the model for testing loss
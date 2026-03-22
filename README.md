# Word2Vec from Scratch (NumPy)

From-scratch implementation and analysis of Word2Vec training dynamics:
- Skip-Gram + Negative Sampling
- CBOW + Negative Sampling
- Manual forward pass, loss, gradients, and parameter updates
- No PyTorch / TensorFlow / autograd

## Research Focus

This project investigates how distributional representations emerge under
Word2Vec training dynamics, specifically:
- How hyperparameters (window size, negative samples, embedding dimension)
  affect embedding geometry
- Differences in learning behavior between CBOW and Skip-Gram
- How word frequency impacts representation quality and update dynamics

## Project Structure

- data/raw: downloaded corpus
- data/processed: tokenized data, vocab maps, training samples
- src/data.py: preprocessing + pair/sample generation
- src/word2vec.py: core model (`W_in`, `W_out`, forward/backward)
- src/train.py: training loops + checkpoints
- src/experiments.py: compact experiment suite
- src/analysis.py: neighbors, analogy helpers, freq-vs-rare stats
- src/evaluate.py: generates evaluation JSON
- src/visualize.py: PCA/t-SNE plots
- results/logs: loss curves/config/results JSON
- results/plots: PCA and t-SNE images

## Dataset

- Tiny Shakespeare (small clean text corpus)

## What is implemented

1. Data prep:
   - text normalization + tokenization
   - vocab `word -> index` and `index -> word`
   - Skip-Gram pairs `(center -> context)`
   - CBOW samples `(context -> center)`
2. Model setup:
   - `W_in` shape `(V, D)`
   - `W_out` shape `(V, D)`
3. Forward pass:
   - Skip-Gram: center embedding lookup
   - CBOW: context embedding mean
4. Loss:
   - Negative Sampling loss with 1 positive + K negatives
   - Manual stable sigmoid
5. Backprop:
   - Manual gradients for center/context vectors + output embeddings
   - SGD updates without autograd
6. Experiments:
   - window size effect
   - embedding dimension effect
   - negative sample count effect
   - frequent vs rare word embedding norm check
7. Representation analysis:
   - nearest neighbors
   - analogy checks
   - PCA + t-SNE plots

## Quick Run

1) Prepare data

python src/data.py --input data/raw/tiny_shakespeare.txt --output_dir data/processed --window_size 2 --min_count 2 --max_vocab_size 8000

2) Train Skip-Gram

python src/train.py --processed_dir data/processed --mode skipgram --embedding_dim 75 --neg_samples 5 --epochs 3 --learning_rate 0.05 --max_samples_per_epoch 30000 --output_dir results/logs

3) Train CBOW

python src/train.py --processed_dir data/processed --mode cbow --embedding_dim 75 --neg_samples 5 --epochs 3 --learning_rate 0.05 --max_samples_per_epoch 30000 --output_dir results/logs

4) Run compact experiments

python src/experiments.py --processed_dir data/processed --mode skipgram --epochs 1 --learning_rate 0.03 --max_samples_per_epoch 2000 --out_file results/logs/experiments_summary.json

5) Evaluate + visualize

python src/evaluate.py --processed_dir data/processed --embeddings results/logs/cbow_d75_neg5_lr0.05_ep3/W_combined.npy --out_file results/logs/evaluation_cbow_d75.json
python src/visualize.py --processed_dir data/processed --embeddings results/logs/cbow_d75_neg5_lr0.05_ep3/W_combined.npy --top_words 120 --out_dir results/plots

## Sample Training Trend

- Skip-Gram (D=75, K=5):
  - 4.1084 -> 3.6149 -> 3.2665
- CBOW (D=75, K=5):
  - 4.0795 -> 3.3534 -> 2.9971

Loss decreases across epochs for both models.

## Key Insights

- Negative sampling introduces a scaling effect on loss values, making raw loss comparisons across different numbers of negatives non-trivial.
- Frequent words receive more gradient updates, leading to larger embedding norms and more stable representations.
- CBOW converges faster due to aggregated context signals, while Skip-Gram provides stronger learning signals for rare words given sufficient training.
- Embedding quality is fundamentally limited by corpus size; small datasets restrict the emergence of rich semantic structure regardless of model correctness.

## Limitations

- Small corpus size limits semantic richness of learned embeddings.
- No subsampling of frequent words, which can bias gradient updates.
- Negative sampling distribution is not tuned (e.g., unigram^0.75).
- Short training horizon limits analogy and clustering quality.

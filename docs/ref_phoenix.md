# `prismpyp visualize_phoenix`

## Purpose
Launch an **interactive Phoenix web application** to visualize learned embeddings stored in a `.parquet.zip` file.

This command provides an intuitive 2D or 3D embedding visualization interface for inspecting latent feature distributions produced by prismPYP or related models.

## Usage
```bash
usage: prismpyp visualize_phoenix parquet_file [--port PORT] [--which-embedding {pca,umap,tsne}]
```

## Positional Argument
| Argument | Description |
|-----------|--------------|
| `parquet_file` | Path to `.parquet.zip` file containing embeddings |

## Named Arguments
| Argument | Description | Default |
|-----------|--------------|----------|
| `--port` | Port number to run the Phoenix server on | `5000` |
| `--which-embedding` | Which embedding space to visualize (`pca`, `umap`, or `tsne`) | `umap` |


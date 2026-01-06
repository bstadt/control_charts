"""Setup script for downloading and embedding NQ dataset."""

import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_CACHE_PATH = Path(__file__).parent.parent.parent.parent / "data" / "nq_embedded.parquet"


def download_nq_dataset(max_samples: int | None = None) -> pd.DataFrame:
    """Download Natural Questions dataset from HuggingFace."""
    from datasets import load_dataset

    logger.info("Downloading Natural Questions dataset from HuggingFace...")
    dataset = load_dataset("sentence-transformers/natural-questions", split="train")

    df = pd.DataFrame({
        "question": dataset["query"],
        "answer": dataset["answer"]
    })

    if max_samples is not None:
        df = df.head(max_samples)

    logger.info(f"Downloaded {len(df)} QA pairs")
    return df


def save_cache(df: pd.DataFrame, embeddings: np.ndarray, path: Path) -> None:
    """Save embedded QA pairs to parquet cache."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert embeddings to list for parquet storage
    df = df.copy()
    df["embedding"] = list(embeddings)

    df.to_parquet(path, index=False)
    logger.info(f"Saved {len(df)} embedded QA pairs to {path}")


def load_cache(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Load embedded QA pairs from parquet cache."""
    df = pd.read_parquet(path)
    embeddings = np.vstack(df["embedding"].values)
    return df[["question", "answer"]], embeddings


@click.command()
@click.option("--remote/--local", default=True, help="Use Modal (remote) or local embedding")
@click.option("--max-samples", type=int, default=None, help="Limit number of samples (for testing)")
@click.option("--output", type=click.Path(), default=None, help="Output parquet path")
@click.option("--batch-size", type=int, default=256, help="Embedding batch size")
def main(remote: bool, max_samples: int | None, output: str | None, batch_size: int):
    """Download and embed Natural Questions dataset."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    output_path = Path(output) if output else DEFAULT_CACHE_PATH

    # Download dataset
    df = download_nq_dataset(max_samples=max_samples)

    # Embed questions
    questions = df["question"].tolist()
    if remote:
        from .embedding import embed_remote
        embeddings = embed_remote(questions, batch_size=batch_size)
    else:
        from .embedding import embed_local
        embeddings = embed_local(questions, batch_size=batch_size)

    # Save cache
    save_cache(df, embeddings, output_path)

    click.echo(f"\n✅ Setup complete! Embedded {len(df)} QA pairs to {output_path}")


if __name__ == "__main__":
    main()

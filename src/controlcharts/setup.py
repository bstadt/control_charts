"""Setup script for downloading and embedding NQ dataset using Modal."""

import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Modal configuration
MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
MODEL_REVISION = "d802ae16c9caed4d197895d27c6d529434cd8c6d"
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


def embed_local(questions: list[str], batch_size: int = 64) -> np.ndarray:
    """Embed questions using local model (slower but no Modal required)."""
    from sentence_transformers import SentenceTransformer

    logger.info(f"Loading model {MODEL_ID} locally...")
    model = SentenceTransformer(MODEL_ID, trust_remote_code=True)

    logger.info(f"Embedding {len(questions)} questions locally...")
    embeddings = model.encode(
        questions,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=False,
        convert_to_numpy=True
    )

    return np.asarray(embeddings, dtype=np.float32)


def embed_remote(questions: list[str], batch_size: int = 256) -> np.ndarray:
    """Embed questions using Modal with GPU acceleration."""
    try:
        import modal
    except ImportError:
        raise SystemExit("modal is required for remote embedding. Install with: pip install modal")

    logger.info("Setting up Modal for remote embedding...")

    image = modal.Image.debian_slim().pip_install(
        "torch==2.6.0",
        "sentence-transformers==3.4.1",
        "einops==0.8.1",
        "numpy",
    )
    app = modal.App("controlcharts-embedding", image=image)

    CACHE_DIR = "/cache"
    cache_vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

    @app.cls(
        gpu="H100",
        volumes={CACHE_DIR: cache_vol},
        timeout=60 * 30,
        scaledown_window=60 * 5,
        max_containers=5,
    )
    class EmbeddingModel:
        @modal.enter()
        def setup(self):
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(
                MODEL_ID,
                revision=MODEL_REVISION,
                cache_folder=CACHE_DIR,
                trust_remote_code=True,
            )
            logger.info(f"Modal: Loaded embedding model {MODEL_ID}")

        @modal.method()
        def embed_batch(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=False,
                convert_to_numpy=True,
            )
            return np.asarray(embeddings, dtype=np.float32)

    # Process in chunks for parallelization
    chunk_size = 1000
    chunks = [questions[i:i + chunk_size] for i in range(0, len(questions), chunk_size)]
    logger.info(f"Processing {len(questions)} questions in {len(chunks)} chunks")

    with app.run():
        model = EmbeddingModel()

        # Submit all chunks in parallel
        futures = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Submitting chunk {i + 1}/{len(chunks)} ({len(chunk)} questions)")
            future = model.embed_batch.spawn(chunk, batch_size=batch_size)
            futures.append(future)

        # Collect results
        all_embeddings = []
        for i, future in enumerate(futures):
            logger.info(f"Waiting for chunk {i + 1}/{len(chunks)}...")
            chunk_embeddings = future.get()
            all_embeddings.append(chunk_embeddings)
            logger.info(f"Chunk {i + 1}/{len(chunks)} completed")

    embeddings = np.concatenate(all_embeddings, axis=0)
    logger.info(f"Remote embedding completed. Shape: {embeddings.shape}")
    return embeddings


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
        embeddings = embed_remote(questions, batch_size=batch_size)
    else:
        embeddings = embed_local(questions, batch_size=batch_size)

    # Save cache
    save_cache(df, embeddings, output_path)

    click.echo(f"\n✅ Setup complete! Embedded {len(df)} QA pairs to {output_path}")


if __name__ == "__main__":
    main()

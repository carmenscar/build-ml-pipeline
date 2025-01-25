#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    logging.info("Downloading artifact: %s", args.input_artifact)
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)
    logging.info("Artifact loaded")

    logging.info("Dropping outliers")
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()

    logging.info("Transforming column in date format")
    df['last_review'] = pd.to_datetime(df['last_review'])

    logger.info("Saving df: 'clean_sample.csv'...")
    df.to_csv("clean_sample.csv", index=False)

    logger.info("Uploading %s to W&B...", args.output_artifact)
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)

    logger.info("Atifact %s uploaded!", args.output_artifact)
    run.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="The name of the input artifact in W&B containing the raw data to be cleaned (e.g., 'airbnb_raw_data:latest').",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="The name of the output artifact in W&B where the cleaned data will be saved (e.g., 'airbnb_clean_data').",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="The type of the output artifact (e.g., 'dataset', 'model', etc.) to be used when saving the cleaned data to W&B.",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="A detailed description of the output artifact explaining what the data or model represents (e.g., 'Airbnb data after cleaning, with prices within the specified range').",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="The minimum price value to be kept in the cleaned data. Any entry with a price below this value will be discarded.",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="The maximum price value to be kept in the cleaned data. Any entry with a price above this value will be discarded.",
        required=True
    )


    args = parser.parse_args()

    go(args)

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

    ######################
    # YOUR CODE HERE     #
    ######################

    logger.info("Downloading artifact")
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()

    df = pd.read_csv(artifact_path)

    # drop the outliers
    logger.info("Dropping outliers in the price feature")
    min_price = args.min_price
    max_price = args.max_price

    idx = df['price'].between(min_price, max_price)
    df = df[idx].copy()

    # Convert the last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])

    # another filtering
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    filename = "clean_sample.csv"
    df.to_csv(filename, index=False)

    # Uploading the output to W&B
    logger.info("Logging artifact")
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    
    run.log_artifact(artifact)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Preprocess a dataset: drop outliers, datetime some columns, etc",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Output file from our code",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Type of the output file",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="description",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Min value to be used as threshold for the outlier drop",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Min value to be used as threshold for the outlier drop",
        required=True
    )


    args = parser.parse_args()

    go(args)

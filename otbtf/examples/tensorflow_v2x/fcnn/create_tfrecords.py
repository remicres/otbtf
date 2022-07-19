"""
This example shows how to convert patches-images (like the ones generated from the `PatchesExtraction`)
into TFRecords files.
"""
import argparse
from pathlib import Path
import helper
from otbtf import DatasetFromPatchesImages

# Application parameters
parser = argparse.ArgumentParser(description="Converts patches-images into TFRecords")
parser.add_argument("--xs", required=True, nargs="+", default=[], help="A list of patches-images for the XS image")
parser.add_argument("--labels", required=True, nargs="+", default=[], help="A list of patches-images for the labels")
parser.add_argument("--outdir", required=True, help="Output dir for TFRecords files")
params = parser.parse_args()


if __name__ == "__main__":

    # Sort patches and labels
    patches = sorted(params.patches)
    labels = sorted(params.labels)

    # Check patches and labels are correctly sorted
    helper.check_files_order(patches, labels)

    # Create output directory
    outdir = Path(params.outdir)
    if not outdir.exists():
        outdir.mkdir(exist_ok=True)

    # Create dataset from the filename dict
    dataset = DatasetFromPatchesImages(filenames_dict={"input_xs": patches, "labels": labels})

    # Convert the dataset into TFRecords
    dataset.to_tfrecords(output_dir=params.outdir)

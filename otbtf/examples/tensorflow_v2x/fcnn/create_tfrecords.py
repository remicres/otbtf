"""
This example shows how to convert patches-images (like the ones generated from the `PatchesExtraction`)
into TFRecords files.
"""
import argparse
from pathlib import Path
from otbtf.examples.tensorflow_v2x.fcnn import helper
from otbtf import DatasetFromPatchesImages

parser = argparse.ArgumentParser(description="Converts patches-images into TFRecords")
parser.add_argument("--xs", required=True, nargs="+", default=[], help="A list of patches-images for the XS image")
parser.add_argument("--labels", required=True, nargs="+", default=[],
                    help="A list of patches-images for the labels")
parser.add_argument("--outdir", required=True, help="Output dir for TFRecords files")


def create_tfrecords(params):
    # Sort patches and labels
    patches = sorted(params.xs)
    labels = sorted(params.labels)

    # Check patches and labels are correctly sorted
    helper.check_files_order(patches, labels)

    # Create output directory
    outdir = Path(params.outdir)
    if not outdir.exists():
        outdir.mkdir(exist_ok=True)

    # Create dataset from the filename dict
    dataset = DatasetFromPatchesImages(filenames_dict={"input_xs_patches": patches, "labels_patches": labels})

    # Convert the dataset into TFRecords
    dataset.to_tfrecords(output_dir=params.outdir, drop_remainder=False)


if __name__ == "__main__":
    params = parser.parse_args()
    create_tfrecords(params)

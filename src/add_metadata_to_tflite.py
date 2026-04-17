from tflite_support import metadata as _metadata
from tflite_support import metadata_schema_py_generated as _metadata_fb
from pathlib import Path
import traceback
import flatbuffers


def main():
    tflite_path = "src/tflite/model.tflite"
    labels_path = Path("src/tflite/labels.txt")
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    print("\nAttaching metadata...")
    try:
        attach_metadata(tflite_path, labels_path)
    except Exception as e:
        print(f"Metadata step failed: {e}")
        traceback.print_exc()


def attach_metadata(tflite_path: Path, labels_path: Path):
    model_meta = _metadata_fb.ModelMetadataT()
    model_meta.name = "EfficientNet Lite4 Classifier"
    model_meta.version = "1.0"

    input_meta = _metadata_fb.TensorMetadataT()
    input_meta.name = "image"
    input_meta.description = "Input RGB image (H x W x 3, float32)"

    # MediaPipe REQUIRES these for float32, but since your model
    # handles normalization internally, we use "Identity" values:
    norm_unit = _metadata_fb.ProcessUnitT()
    norm_unit.optionsType = _metadata_fb.ProcessUnitOptions.NormalizationOptions
    norm_opts = _metadata_fb.NormalizationOptionsT()
    norm_opts.mean = [0.0, 0.0, 0.0]
    norm_opts.std = [1.0, 1.0, 1.0]
    norm_unit.options = norm_opts
    input_meta.processUnits = [norm_unit]

    output_meta = _metadata_fb.TensorMetadataT()
    output_meta.name = "probability"
    output_meta.description = "Per-class softmax scores"

    label_file = _metadata_fb.AssociatedFileT()
    label_file.name = labels_path.name
    label_file.description = "Class labels"
    label_file.type = _metadata_fb.AssociatedFileType.TENSOR_AXIS_LABELS
    output_meta.associatedFiles = [label_file]

    subgraph = _metadata_fb.SubGraphMetadataT()
    subgraph.inputTensorMetadata = [input_meta]
    subgraph.outputTensorMetadata = [output_meta]
    model_meta.subgraphMetadata = [subgraph]

    builder = flatbuffers.Builder(0)
    builder.Finish(
        model_meta.Pack(builder), _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER
    )
    metadata_buf = builder.Output()

    populator = _metadata.MetadataPopulator.with_model_file(str(tflite_path))
    populator.load_metadata_buffer(metadata_buf)
    populator.load_associated_files([str(labels_path)])
    populator.populate()

    print(f"Metadata attached successfully to: {tflite_path}")


if __name__ == "__main__":
    main()

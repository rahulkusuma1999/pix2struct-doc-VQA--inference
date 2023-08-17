# Benchmarking Results

|Model | Inference time | Model size|
|------|----------------|-----------|
 huggingface model |31.46s |1129.27 MB|
 encoder_model.onnx|30.82s|367.55 MB|
 decoder_model.onnx |30.01s|762.17 MB|
 decoder_with_past_model.onnx|29.79s|705.52 MB|
 decoder_model_merged.onnx|28.79s|762.52 MB|
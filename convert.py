#please install optimum and transformers

from transformers import Pix2StructForConditionalGeneration

model_name = "google/pix2struct-docvqa-base"
model = Pix2StructForConditionalGeneration.from_pretrained(model_name)

#CLI command to convert hugging face model to ONNX.

optimum-cli export onnx  --model google/pix2struct-docvqa-base pix2struct_onnx


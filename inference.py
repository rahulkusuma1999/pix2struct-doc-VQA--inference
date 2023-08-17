
import requests
from PIL import Image
from transformers import Pix2StructForConditionalGeneration,Pix2StructProcessor
from functools import partial
import time
import onnxruntime as ort
import onnx
import torch
import numpy as np
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", required=True, help="Path to the ONNX model folder.")
    parser.add_argument("-i", "--image_path", required=True, help="Path to the input image.")
    args = parser.parse_args()

    # loading model

    model_name = "google/pix2struct-docvqa-base"
    model = Pix2StructForConditionalGeneration.from_pretrained(model_name)
    processor = Pix2StructProcessor.from_pretrained("google/pix2struct-docvqa-base")

    
    image = Image.open(args.image_path)
    onnx_model = ort.InferenceSession(args.model_path)
    questions = input("question to ask regarding document or image : ")

    # inference for huggingface original model

    def generate(model, processor, img, questions):
      inputs = processor(images=[img for _ in range(len(questions))],
          text=questions, return_tensors="pt")
      predictions = model.generate(**inputs, max_new_tokens=256)
      return zip(questions, processor.batch_decode(predictions, skip_special_tokens=True))

    
    start_time = time.time()
    generator = partial(generate, model, processor)
    completions = generator(image, questions)
    for completion in completions:
      print(f"{completion}")
      end_time = time.time()
      inference_time = end_time - start_time
      print(f"{inference_time} seconds")



    
    # inference fo ONNX model

    def generate(onnx_model, processor, img, questions):
      inputs = processor(images=[img for _ in range(len(questions))],
          text=questions, return_tensors="pt")
      predictions = model.generate(**inputs, max_new_tokens=256)
      return zip(questions, processor.batch_decode(predictions, skip_special_tokens=True))


    start_time = time.time()
    generator = partial(generate, onnx_model, processor)
    completions = generator(image, questions)
    for completion in completions:
      print(f"{completion}")
    end_time = time.time()
    inference_time = end_time - start_time
    print(f"{inference_time} seconds")







    model_path = "model_state_dict.pth"
    torch.save(model.state_dict(), model_path)

    # Get the size of the saved model file
    model_size = os.path.getsize(model_path)

    print(f"Model size of original huggingface model: {model_size/1e6} MB ")

    print(f"model size of ONNX model{os.path.getsize(args.model_path)/1e6} MB")





if __name__ == "__main__":
  main()

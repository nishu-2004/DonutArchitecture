import os
import re
import json
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import google.generativeai as genai

# Configure Gemini API key
genai.configure(api_key='YOUR_API_KEY')

def process_image_with_donut(image_path, output_folder, task_prompt, model_name):
    """
    Process an image using the Donut model and save the output in a specified folder.

    Args:
        image_path (str): Path to the input image file.
        output_folder (str): Path to the folder where the output will be saved.
        task_prompt (str): Task-specific prompt for the Donut model.
        model_name (str): Donut model name to use.

    Returns:
        str: Path to the generated .json file.
    """
    # Load the Donut model and processor
    processor = DonutProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    # Prepare decoder input
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

    # Generate output
    outputs = model.generate(
        pixel_values=pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=model.decoder.config.max_position_embeddings,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    # Postprocess output
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()
    output_json = processor.token2json(sequence)

    # Save output to the specified folder
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, os.path.basename(image_path).split(".")[0] + ".json")

    with open(output_file, "w") as f:
        json.dump(output_json, f, indent=4)

    print(f"Processed output saved to: {output_file}")
    return output_file

def use_json_as_gemini_prompt(json_file_path):
    """
    Use the content of a .json file as the prompt for the Gemini API.

    Args:
        json_file_path (str): Path to the .json file.
    """
    # Load the .json file content
    with open(json_file_path, 'r') as file:
        file_content = json.load(file)

    # Create Gemini generation configuration
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    # Initialize the Gemini model
    model = genai.GenerativeModel(
        model_name="learnlm-1.5-pro-experimental",
        generation_config=generation_config,
    )

    # Start a chat session
    chat_session = model.start_chat(history=[])

    # Send the .json content as the prompt
    response = chat_session.send_message(f"explain the image based on this{json.dumps(file_content)}, Dont add class . Just explain what the image is.")

    # Print Gemini response
    print("Gemini Response:")
    print(response.text)

if __name__ == "__main__":
    # Input image path
    input_image_path = input("Enter the path to the input image: ")

    # Output folder
    output_folder = input("Enter the path to the output folder: ")

    # Try processing the image with each model and task prompt
    try:
        print("Trying to process the image using text-based task (RVLCDIP)...")
        json_file_path = process_image_with_donut(
            input_image_path,
            output_folder,
            task_prompt="<s_rvlcdip>",
            model_name="naver-clova-ix/donut-base-finetuned-rvlcdip"
        )
    except Exception as e:
        print("Text-based processing failed, switching to structured data task (CORD)...")
        json_file_path = process_image_with_donut(
            input_image_path,
            output_folder,
            task_prompt="<s_cord-v2>",
            model_name="naver-clova-ix/donut-base-finetuned-cord-v2"
        )

    # Use the .json file as the prompt for Gemini
    use_json_as_gemini_prompt(json_file_path)


 **Image Processing and Description Generation using Donut and Gemini**  

This code leverages **Donut (Document Understanding Transformer)** models to extract structured information from images and then utilizes **Google Gemini** to generate a detailed textual description based on the extracted data.  

 **Models Used**  
1. **`naver-clova-ix/donut-base-finetuned-rvlcdip`**  
   - Fine-tuned on the **RVLCDIP dataset** for document classification and text-based information extraction.  

2. **`naver-clova-ix/donut-base-finetuned-cord-v2`**  
   - Fine-tuned on the **CORD dataset**, designed for structured data extraction (e.g., receipts and key-value pairs).  
   - Used as a fallback if the text-based model fails.  

 **Workflow**  
1. The program takes an **input image** and processes it using the Donut model.  
2. The extracted information is **saved as a `.json` file** in the specified output directory.  
3. The content of the generated JSON file is used as a **prompt** for Google Gemini.  
4. Gemini generates a **detailed textual description** of the image based on the extracted data.  

**Key Features**  
- **Automatic model selection**: If `RVLCDIP` processing fails, the script switches to the `CORD` model.  
- **JSON-based structured output**: The extracted data is stored in a machine-readable format.  
- **Natural language explanation**: Gemini interprets the JSON content to generate a human-readable description.  

This approach enables **automated document understanding** and **image-to-text conversion**, making it useful for applications like document analysis, automated metadata generation, and image-based search indexing.

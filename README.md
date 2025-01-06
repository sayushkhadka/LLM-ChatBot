# Chatbot with Conversational Forms 

## Overview  
This project implements a chatbot that can:  
1. Answer user queries from uploaded documents using LangChain and LLM.  
2. Collect user information (Name, Phone Number, Email) through a conversational form when the user requests a callback.  
3. Book appointments using a conversational form integrated with LLM tool agents.  
4. Extract and validate date formats (e.g., `YYYY-MM-DD`, next Monday, Monday, today, tomorrow) and user inputs like email and phone numbers.  

## Features  
- **Document Querying:** Users can upload documents, and the chatbot answers their questions based on the content.  
- **Conversational Forms:** Collects user details for callbacks and appointment bookings.  
- **Date Parsing:** Extracts complete date formats from natural language queries.  
- **Input Validation:** Ensures correct formatting for user-provided data (e.g., name, email, phone number).  

## Technologies Used  
- **LangChain**  
- **Hugging Face for the LLM models:**  
  - `HuggingFaceH4/zephyr-7b-beta` --> text generation  
  - `microsoft/Phi-3-mini-4k-instruct --> text-generation`  
  - `google/flan-t5-large` --> text-to-text generation  
  - `sentence-transformers/all-MiniLM-L6-v2` --> transformer embeddings  
- **Python**  
- **Tool Agents**  

## How to Run  
1. Ensure all dependencies are installed: pip install -r requirements.txt
2. Run the `main.py` file for overall functionalities: python main.py 
    

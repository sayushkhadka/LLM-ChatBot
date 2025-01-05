from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from pydantic import BaseModel, Field

from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_unstructured import UnstructuredLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

import logging
import os

import re

#REQUIRES HUGGING FACE API KEY


hf_api_token = os.getenv("HF_API_TOKEN")
# setting the logging level to ERROR to suppress all logs except errors
logging.basicConfig(level=logging.ERROR)

#loading the environment variables from .env
load_dotenv()

#handles user personal info --> it is for 
class PersonalDetails(BaseModel):
    full_name: str = Field(None, description="The full name of the user.")
    number: str = Field(None, description="The phone number of the user.")
    email: str = Field(None, description="An email address that the person associates as theirs.")
    call: str = Field(None, description="User requests to call.")

#class that handles the overall chatbot scenario
class Chatbot:
    def __init__(self, model_repo="microsoft/Phi-3-mini-4k-instruct"):

        #initializing model endpoint --> hugging face model
        #model below is huggingface chat generation model
        self.llm = HuggingFaceEndpoint(
            repo_id=model_repo,  # model repo
            task="text-generation",  #task for the model
            max_new_tokens=512,  #max tokens to generate
            do_sample=False,  #whether to sample the output
            repetition_penalty=1.03,  # penalty for repetition
            huggingfacehub_api_token = hf_api_token
        )
        self.chat = ChatHuggingFace(llm=self.llm, verbose=True) # creates a chat interface using the initialized model
        self.chain = self.chat.bind_tools([PersonalDetails]) #bind the tools

    def process_input(self, user_input):
        try:
            #invoke the chat chain with the use input
            res = self.chain.invoke(user_input)

            #checking if the tool was called here ----> PersonalDetails
            if hasattr(res, 'tool_calls') and res.tool_calls:
                first_tool_call = res.tool_calls[0]
                personal_details = first_tool_call['args']

                #checking if personal details were provided by the user
                details_found = False
                if personal_details['full_name']:
                    details_found = True
                if personal_details['number']:
                    details_found = True
                if personal_details['email']:
                    details_found = True

                #to print out the extracted details
                if details_found:
                    print(f"{'-'*170}")
                    print("ChatBot: ")
                    print("Information received as follows:")
                    print(f"Full Name: {personal_details['full_name']}")
                    print(f"Phone Number: {personal_details['number']}")
                    print(f"Email: {personal_details['email']}")
                    print("If any of this information is incorrect, please contact us.")
                    print(f"{'-'*170}")
                else:
                    print("ChatBot: No personal details found. Please enter your name, email, and phone number.")
                    print(f"{'-'*170}")
            else:
                print("ChatBot: No personal details found. Please enter your name, email, and phone number.")
                print(f"{'-'*170}")
    
        except KeyError as e:
            print(f"ChatBot: Missing information: {e}. Please enter all required details again.")
        except Exception as e:
            print(f"ChatBot: An error occurred: {e}")


#handle the document provided by the user
class DocumentHandler:
    def __init__(self):
        pass

    #function that loads the document
    def load_document(self, path):
        try:
            file_loader = UnstructuredLoader(path) #unstructuredloader allows different file type
            documents = file_loader.load()
            return documents
        except Exception as e:
            print(f"Error loading document: {e}")
            return None

    #dividing the document to chunks for processing ease for the model
    def chunk_data(self, docs, chunk_size=400, chunk_overlap=50):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        doc_chunks = text_splitter.split_documents(docs)
        return doc_chunks

   #vector embeddings for the chunks 
    def embeddings_creation(self, documents):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") # the embedding model
        vector = FAISS.from_documents(documents, embeddings)  # --> DB for vector
        return vector

    #searches for the top match for k matching chunks based on the query.
    def retrieve_query(self, query, vector, k=2):  # --> k=2; looks for top 2 query
        matching_results = vector.similarity_search(query, k=k) # searches for similar embeddings or vector of the chunks
        return matching_results

    #this is the QA chain to generate the answer for user query from the entered document
    def retrieve_answers(self, query, qa_chain):
        result = qa_chain.invoke(query)
        print(f"You: {query}")
        print(f"Chatbot: {result['result']}")
        print(f"{'-'*170}")

#this executes the handled documents and reads
class DocumentReader:
    def __init__(self, chatbot, document_handler):
        self.chatbot = chatbot
        self.document_handler = document_handler

    def document_reader(self):
        """
        Main function to load a document, process it
        """
        path = input("Please enter the path to the document:")  
        # input should be (documents\filename) , for e.g. --->  documents\MLPDF.pdf
        #if this does not work enter the absolute path
        print()
        documents = self.document_handler.load_document(path)

        if documents:
            #chunk the document into smaller pieces
            chunked_documents = self.document_handler.chunk_data(documents)

            #creating embeddings from the chunked documents
            vectorstore = self.document_handler.embeddings_creation(chunked_documents)

            #different model has been used from the chat generation model
            model_name = "google/flan-t5-large"
            hf_pipeline = pipeline("text2text-generation", model=model_name)
            llm = HuggingFacePipeline(pipeline=hf_pipeline)

            if llm:
                #initializing the RetrievalQA chain with the vectorstore---> DB for our embeddings and LLM ---> model initializing
                qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

                print()
                print("Document Reader is ready. You can ask questions about the document.")
                print("Type 'exit' to quit.")
                while True:
                    query = input("Please ask me about the document: ")
                    if query.lower() == "exit":
                        print("Exiting the document reader. Goodbye!")
                        break
                    else:
                        self.document_handler.retrieve_answers(query, qa_chain)
            else:
                print("LLM initialization failed.")
        else:
            print("Failed to load document.")

#the main class that integrates the functionalities
class MainChat:
    def __init__(self):
        self.chatbot = Chatbot()
        self.document_handler = DocumentHandler()
        self.document_reader = DocumentReader(self.chatbot, self.document_handler)
        self.user_details = {}  #stores user details here

    #this gets executed when the user inputs "call"
    def collect_personal_details(self):
        print(f"{'-'*170}")
    #collecting and validating full name
        while True:
            print("ChatBot: Please provide your full name:")
            name = input("You: ").strip()
            if name.lower() == 'exit':
                print("Exiting the chat. Goodbye!")
                exit()
            elif name:
                self.user_details["full_name"] = name.capitalize()
                break
            else:
                print("ChatBot: Name cannot be empty. Please try again.")

        #collect and validating phone number
        while True:
            print("ChatBot: Please provide your phone number:")
            phone = input("You: ").strip()

            if phone.lower() == 'exit':
                print("Exiting the chat. Goodbye!")
                exit()

            elif re.match(r"^\+?\d{10,15}$", phone):
                self.user_details["number"] = phone
                break
            else:
                print("ChatBot: Phone number must contain 10-15 digits and can include an optional '+' at the start. Please try again.")

        #collect and validating email address
        while True:
            print("ChatBot: Please provide your email address:")
            email = input("You: ").strip()

            if email.lower() == 'exit':
                print("Exiting the chat. Goodbye!")
                exit()

            elif re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", email):
                self.user_details["email"] = email
                break
            else:
                print("ChatBot: Email address must be in a valid format. Please try again.")

    #the start of the chatbot 
    def start_chat(self):
        #for collecting and processing the user's details
        print("Hello! This is your document reader. Type 'exit' to end the conversation.")
        print("ChatBot: Please enter your name, email, and phone number: ")

        while True:
            user_input = input("You: ")

            #exits the chat if the user types 'exit'
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break

            #checks if the user requests a call
            if 'call' in user_input.lower():
                self.collect_personal_details()

                #after collecting detail it allows the user to upload a document
                print(f"{'-'*170}")
                print("ChatBot: Thank you for providing your information. Now you can upload a document.")
                self.document_reader.document_reader()
                break  #exit after document upload

            #process the user input only if it's likely to contain personal details
            #this value gets store when the user doesn't enter 'call'
            #this is more of conversation form
            #the user inputs My name is {name} and my email is {email} and my number is {number}
            #the chat generation model then starts to look for relevant information that was binded with chat.bind_tools
            #we are only using PersonalDetails as a tool and it has {name} {email} {number}
            #so the model specifically looks for these values 
            elif 'name' in user_input.lower() or 'email' in user_input.lower() or 'phone' in user_input.lower():
                self.chatbot.process_input(user_input)

                #if details are received it allows the user to upload a document
                print("ChatBot: Thank you for providing your information. Now you can upload a document.")
                self.document_reader.document_reader()
                break  #exit after document upload is uploaded

            else:
                print("ChatBot: No personal details detected. Please enter your name, email, and phone number.")
                print(f"{'-'*170}")

#entry point for the application
#it is kind of a constructor
#only executed when run directly and not an imported module in other scripts
#this ensures that when imported to other scripts or files, it won't execute (the code inside if __name__ = "__main__") just by import chatbot 
# but we can still access the functions or methods in it like chatbot.MainChat()
# it will be like a module and module named chatbot
if __name__ == "__main__":
    main_chat = MainChat() #instance of the class, main_chat

    main_chat.start_chat() #starts with the start_chat() function stored here in main_chat object

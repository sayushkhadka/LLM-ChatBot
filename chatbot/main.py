from langchain.agents import Tool, initialize_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

import os

import subprocess
import sys

#Start with --------------->  pip install -r requirements.txt   -----------------> to download all the requirements for the scripts
# !!! Downloading scripts might take some time. !!!
# !!! If it takes to much of time, then upgrade pip to the latest version and that should work !!!

# Hugging Face models have been used.
# !!! REQUIRED to create Hugging Face API TOKEN KEY from the Hugging Face website !!!

#Run 'main.py' for overall functionality of the chatbot 

class ConversationalAgent:
    def __init__(self):
        #sets up the hf model and the chat model model for text generation
        self.llm = HuggingFaceEndpoint(
            repo_id="HuggingFaceH4/zephyr-7b-beta",
            task="text-generation",
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.03,
        )
        #initialize the chat model with the hf model
        self.chat_model = ChatHuggingFace(llm=self.llm)

        #setting up memory for the conversational agent
        self.memory = ConversationBufferWindowMemory(
            memory_key='chat_history',
            k=3,
            return_messages=True
        )

        #tools for invoking different functionalities
        self.option_tool_func = Tool(
            name='Option to choose',
            func=self.option_tool,
            description="Useful for when you need to choose the option read document or book appointment. Input should be read or document or book or appointment."
        )

        #conversational agent initialization
        self.conversational_agent = initialize_agent(
            agent='chat-conversational-react-description',  #agent type
            tools=[self.option_tool_func],  #list of tools the agent can use
            llm=self.llm,  #language model for generating responses
            verbose=True,  #enable verbose logging for debugging
            max_iterations=3,  #maximum numer of iterations the agent can run
            early_stopping_method='generate',  #stop early if the response is generated
            memory=self.memory,  #the defined memory for the conversation
            handle_parsing_errors=True  #hndle errors during parsing
        )


        #the custom prompt provided to the agent
        self.conversational_agent.agent.llm_chain.prompt.messages[0].prompt.template = '''Assistant is a large language model trained by OpenAI.

        Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

        Assistant doesn't know anything about random numbers or anything related to the meaning of life and should use a tool for questions about these topics.

        Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

        Overall, Assistant is a powerful system that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.'''

    #tool to select option for the agent (e.g., document reading, appointment booking)
    def option_tool(self, input):
        try:
            #if the user is asking for document reading
            if "document" in input.lower() or "file" in input.lower():
                print("\n[INFO] Running Document Reader...\n")
                subprocess.run([sys.executable, "document_reader.py"], check=True) #runs the document reader
                return "\n[SUCCESS] Document Reader has been executed successfully.\n"
            
            #if the user is asking for appointment booking
            elif "appointment" in input.lower() or "booking" in input.lower() or 'schedule' in input.lower():
                print("\n[INFO] Running Appointment Booking...\n")
                subprocess.run([sys.executable, "appointment_booking.py"], check=True) #runs the appointment_booking
                return "\n[SUCCESS] Appointment Booking system has been executed successfully.\n"
            
            #ifneither, use the chat model for general queries
            else:
            #the chat model for other queries
                response = self.chat_model.invoke(input)
                if isinstance(response, dict) and 'content' in response:
                    return f"ChatBot: {response['content']}"  #extracts and returns the message content
                elif hasattr(response, 'content'):
                    return f"ChatBot: {response.content}"  #for object responses
                else:
                    return f"ChatBot: {str(response)}" 
                
        except subprocess.CalledProcessError as e:
            #handle errors during script execution
            return f"\n[ERROR] Error executing script: {e}\n"
        except Exception as e:
            return f"\n[ERROR] An unexpected error occurred: {e}\n"
        
    #to interact with the user and run the chatbot
    def interact(self):
        print()
        print("You can type 'exit' anytime to quit the chat.\n")
        print(f"{'*' * 70} Welcome to the ChatBot! {'*' * 70}")
        print()
        print("** User Manual**  \nThe chatbot can answer any of your queries.")
        print("** If you wish to use specific functions like: Document Reader or Appointment Booking, mention them and the chatbot will provide you with the service. **")
        print()
        print("Hello User!!! Get started with asking your queries...")
        while True:
            print(f"{'-'*170}")
            user_input = input("You: ")
            
            if user_input.lower() == "exit":
                print("\nThank you for using the chatbot. Goodbye!")
                break
            
            result = agent.option_tool(user_input)
            print(result)

        #the input_text to the option_tool method
        #return self.option_tool(user_input)

#runs the class when the script is execuetd
if __name__ == "__main__":
    agent = ConversationalAgent() #initializing the agent

    agent.interact()#start the interaction loop
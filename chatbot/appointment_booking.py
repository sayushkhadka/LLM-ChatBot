from langchain.agents import Tool, initialize_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

import os
import re

from datetime import datetime, timedelta
from dateutil import parser

class AppointmentBooking:
    def __init__(self):
        #initializing instance variables
        self.book_appointment_tool = None
        self.tools = None
        self.memory = None
        self.hf_api_token = None
        self.llm = None
        self.chat_model = None
        self.conversational_agent = None

        #defining a fixed prompt for the conversational agent
        self.fixed_prompt = '''Assistant is a large language model trained by HuggingFace. 
        Assistant is designed to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on various topics. As a language model, Assistant can generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.
        Assistant doesn't know anything about random numbers or anything related to the booking appointment tool and should use a tool for questions about these topics.
        Assistant is constantly learning and improving, and its capabilities are constantly evolving, processing and understanding large amounts of text. 
        Overall, Assistant is a powerful system that can help with a wide range of tasks and provide valuable insights and information on various topics.'''

    def setup(self):
        #initializing the book appointment tool
        self.book_appointment_tool = Tool(
            name='Book Appointment',
            func=self.book_appointment,
            description="Useful for when you need to book an appointment. Input should be a date and time."
        )
        
        #tools to be used
        self.tools = [self.book_appointment_tool]

        #initialize memory to store conversation history
        self.memory = ConversationBufferWindowMemory(
            memory_key='chat_history',
            k=3,
            return_messages=True
        )

        #initializing HuggingFace API token
        self.hf_api_token = os.getenv("HF_API_TOKEN")

        #hf model for text generation
        self.llm = HuggingFaceEndpoint(
            repo_id="HuggingFaceH4/zephyr-7b-beta",
            task="text-generation",
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.03,
        )

        #creating a chat model based on HuggingFace
        self.chat_model = ChatHuggingFace(llm=self.llm)

        #initializing the conversational agent with tools, model, and memory
        self.conversational_agent = initialize_agent(
            agent='chat-conversational-react-description',
            tools=self.tools,
            llm=self.chat_model,
            verbose=True,
            max_iterations=3,
            early_stopping_method='generate',
            memory=self.memory,
            handle_parsing_errors=True
        )

        #the fixed prompt for the agent's behavior
        self.conversational_agent.agent.llm_chain.prompt.messages[0].prompt.template = self.fixed_prompt

    #main function to book appointment including validation 
    def book_appointment(self, name, email, phone, date, time):
        try:
            #parse user-entered date
            #user entered date
            #for if user enters today for day
            if date.lower() == 'today': #handles 'today' as a special case
                parsed_date = datetime.today()

            #for if user enters tomorrow for day
            elif date.lower() == 'tomorrow': #handles 'tomorrow' as a special case
                parsed_date = datetime.today() + timedelta(days=1)
            else:
                #parse other date inputs 
                parsed_date = parser.parse(date, fuzzy=True) #helps with days like sunday, monday, next monday, coming monday, etc.

            #for user entered time
            if time:
                try:
                    parsed_time = parser.parse(time).strftime("%H:%M") #hours:mins
                    hour = int(parsed_time.split(":")[0])

                    #appointment can be booked from 10am to 6pm
                    if hour < 10 or hour > 18: 
                        return "Invalid time. Please enter a time between 10 AM and 6 PM."
                except ValueError:
                    return "Invalid time format. Please enter a valid time (e.g., '2 PM', '14:00')."
            else:
                parsed_time = "Not specified"

            #formattings of the date and time
            formatted_date = parsed_date.strftime("%B %d, %Y")
            month_day_year = parsed_date.strftime("%d-%m-%Y")
            return f"Your details are as follows:\n\n{'-'*40}\nName: {name}\nEmail: {email}\nPhone: {phone}\nDate: {formatted_date} ({month_day_year})\nTime: {parsed_time}\n{'-'*40}"
        
        except ValueError:
            return "Invalid date format. Please enter a valid date."

    #email validation 
    def is_valid_email(self, email):
        """Validate email format"""
        return bool(re.match(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$", email))

    #phone number validation 
    def is_valid_phone_number(self, phone):
        """Validate phone number format"""
        return bool(re.match(r"^\+?\d{10,15}$", phone))

    #this is the starting chat section, main function to start booking process
    def start_booking(self):
        print()
        print("You can type 'exit' anytime to quit the chat.\n")
        print("********** Start of Appointment Booking **********\n")

        #start of the conversatio

        name = self.get_name()
        email = self.get_email(name)
        phone = self.get_phone(name)
        date_input = self.get_date(name)
        time_input = self.get_time(name)

        #after gathering all inputs, show the booking details
        response = self.book_appointment(name, email, phone, date_input, time_input)
        print("\n********** Booking Summary **********\n")
        print(response)
        print("\n********** End of Appointment Booking **********")

    #getter for the user's name and name validation
    def get_name(self):
        while True:
            name = input("Enter your name: ").strip()
            if name.lower() == 'exit':
                print("Exiting the chat. Goodbye!")
                exit()
            elif name:
                print(f"Got it, {name}. Let's proceed!")
                return name.capitalize()
            else:
                print("Name cannot be blank. Please enter a valid name.")

    #getter for email and email validation
    def get_email(self, name):
        while True:
            email = input(f"{name}, please enter your email: ")
            if email.lower() == 'exit':
                print("Exiting the chat. Goodbye!")
                exit()
            if self.is_valid_email(email):
                print(f"Thank you, {name}. Your email is {email}.")
                return email
            else:
                print("Invalid email format. Please enter a valid email.")
    #getter for phone and phone validation
    def get_phone(self, name):
        while True:
            phone = input(f"{name}, please enter your phone number: ")
            if phone.lower() == 'exit':
                print("Exiting the chat. Goodbye!")
                exit()
            if self.is_valid_phone_number(phone):
                print(f"Thank you, {name}. Your phone number is {phone}.")
                return phone
            else:
                print("Invalid phone number format. Please enter a valid phone number.")

        #getter for the user's appointment date with validation
    def get_date(self, name):
        while True:
            date_input = input(f"{name}, please enter the date for your appointment (e.g., '01/12/2025' or 'next Monday'): ")
            if date_input.lower() == 'exit':
                print("Exiting the chat. Goodbye!")
                exit()
            try:
                if date_input.lower() in ['today', 'tomorrow']:
                    print(f"Got it, {name}. Your appointment will be on {date_input}.")
                    return date_input
                else:
                    parser.parse(date_input, fuzzy=True)
                    print(f"Got it, {name}. Your appointment will be on {date_input}.")
                    return date_input
            except ValueError:
                print("Invalid date format. Please enter a valid date.")

#getter function for the user's appointment time with validation
    def get_time(self, name):
        while True:
            time_input = input(f"{name}, please enter the time for your appointment (e.g., '14:00' or '2 PM'): ")
            if time_input.lower() == 'exit':
                print("Exiting the chat. Goodbye!")
                exit()
            if re.match(r'^(0[0-9]|1[0-9]|2[0-3])(:[0-5][0-9])?$|^([1-9]|1[0-2])\s?(AM|PM)$', time_input.strip(), re.IGNORECASE):
                try:
                    parsed_time = parser.parse(time_input.strip())
                    if 10 <= parsed_time.hour <= 18:
                        print(f"Got it, {name}. Your appointment time is {time_input}.")
                        return time_input
                    else:
                        print("Invalid time. Please enter a time between 10 AM and 6 PM.")
                except ValueError:
                    print("Invalid time format. Please enter a valid time (e.g., '2 PM', '14:00').")
            else:
                print("Invalid time format. Please enter a valid time (e.g., '2 PM', '14:00').")


if __name__ == "__main__":
    #creating an instance of the AppointmentBooking class and start the booking process
    appointment_booking = AppointmentBooking()

    appointment_booking.setup() 
    appointment_booking.start_booking()

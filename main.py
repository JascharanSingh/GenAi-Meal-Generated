from dotenv import load_dotenv
import os
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
import streamlit as st 
from langchain.chains import LLMChain, SequentialChain

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

# Set up OpenAI LLM
llm = OpenAI(api_key=API_KEY, temperature=0.1)

# Prompt 1: Generate meal from ingredients
prompt_template = PromptTemplate(
    template="Give me an example of a meal that could be made using the following ingredients: {ingredients}",
    input_variables=["ingredients"]
)

# Prompt 2: Re-write meal in gangster style
gangster_template = """
Re-write the meal below in the style of a New York mafia gangster:

{meal}
"""
gangster_prompt_template = PromptTemplate(
    template=gangster_template,
    input_variables=["meal"]
)

# Create individual chains
meal_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="meal", verbose=True)
gangster_chain = LLMChain(llm=llm, prompt=gangster_prompt_template, output_key="final_output", verbose=True)

# Sequentially connect the chains
overall_chain = SequentialChain(
    chains=[meal_chain, gangster_chain],
    input_variables=["ingredients"],
    output_variables=["meal", "final_output"],  # include both outputs
    verbose=True
)

# Streamlit UI
st.title("🍽️ Gangster Meal Generator")
user_prompt = st.text_input("What ingredients do you have?")

if st.button("Generate") and user_prompt:
    with st.spinner("Generating..."):
        output = overall_chain({"ingredients": user_prompt})
        
        # Two-column layout
        col1, col2 = st.columns(2)

        col1.markdown("### 🍴 Meal:")
        col1.write(output["meal"])

        col2.markdown("### 😎 Gangster Style:")
        col2.write(output["final_output"])

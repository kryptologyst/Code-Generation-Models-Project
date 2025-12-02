"""
Interactive Streamlit demo for code generation models.
"""

import streamlit as st
import sys
import os
import yaml
import torch
from typing import List, Dict
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import create_model
from utils import set_seed, get_device, Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Code Generation Demo",
    page_icon="💻",
    layout="wide"
)

# Title
st.title("💻 Code Generation Models Demo")
st.markdown("Generate code from natural language descriptions using state-of-the-art transformer models.")

# Sidebar for configuration
st.sidebar.header("Configuration")

# Model selection
model_type = st.sidebar.selectbox(
    "Model Type",
    ["gpt2", "codet5", "codegen"],
    help="Select the model architecture to use"
)

# Model-specific configurations
if model_type == "gpt2":
    model_name = st.sidebar.selectbox(
        "GPT-2 Model",
        ["gpt2", "gpt2-medium", "gpt2-large"],
        help="Select the GPT-2 model size"
    )
elif model_type == "codet5":
    model_name = st.sidebar.selectbox(
        "CodeT5 Model",
        ["Salesforce/codet5-base", "Salesforce/codet5-small"],
        help="Select the CodeT5 model size"
    )
else:  # codegen
    model_name = st.sidebar.selectbox(
        "CodeGen Model",
        ["Salesforce/codegen-350M-mono", "Salesforce/codegen-2B-mono"],
        help="Select the CodeGen model size"
    )

# Generation parameters
st.sidebar.header("Generation Parameters")

max_length = st.sidebar.slider(
    "Max Length",
    min_value=50,
    max_value=500,
    value=150,
    help="Maximum length of generated code"
)

temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.1,
    max_value=2.0,
    value=0.7,
    step=0.1,
    help="Controls randomness. Lower values make output more deterministic."
)

top_p = st.sidebar.slider(
    "Top-p",
    min_value=0.1,
    max_value=1.0,
    value=0.9,
    step=0.05,
    help="Nucleus sampling parameter. Controls diversity of generated text."
)

top_k = st.sidebar.slider(
    "Top-k",
    min_value=1,
    max_value=100,
    value=50,
    help="Top-k sampling parameter. Limits vocabulary to top-k tokens."
)

num_samples = st.sidebar.slider(
    "Number of Samples",
    min_value=1,
    max_value=5,
    value=1,
    help="Number of code samples to generate per prompt"
)

# Random seed
seed = st.sidebar.number_input(
    "Random Seed",
    min_value=0,
    max_value=10000,
    value=42,
    help="Random seed for reproducible generation"
)

# Load model button
@st.cache_resource
def load_model(model_type: str, model_name: str):
    """Load the specified model."""
    try:
        device = get_device()
        model = create_model(model_type, model_name, device)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Load model
with st.spinner("Loading model..."):
    model = load_model(model_type, model_name)

if model is None:
    st.error("Failed to load model. Please check the configuration.")
    st.stop()

# Main interface
st.header("Code Generation")

# Input prompt
prompt = st.text_area(
    "Enter your prompt:",
    value="Write a Python function that calculates the factorial of a number",
    height=100,
    help="Describe the code you want to generate in natural language"
)

# Generate button
if st.button("Generate Code", type="primary"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating code..."):
            try:
                # Set seed for reproducibility
                set_seed(seed)
                
                # Generate code
                samples = model.generate(
                    prompt=prompt,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    num_return_sequences=num_samples
                )
                
                # Display results
                st.success("Code generated successfully!")
                
                for i, sample in enumerate(samples):
                    st.subheader(f"Generated Code {i+1}")
                    st.code(sample, language="python")
                    
                    # Download button
                    st.download_button(
                        label=f"Download Code {i+1}",
                        data=sample,
                        file_name=f"generated_code_{i+1}.py",
                        mime="text/python"
                    )
                
            except Exception as e:
                st.error(f"Error generating code: {str(e)}")

# Example prompts
st.header("Example Prompts")
example_prompts = [
    "Write a Python function that calculates the factorial of a number",
    "Create a function to reverse a string",
    "Implement a binary search algorithm",
    "Write a function to check if a number is prime",
    "Create a function to find the maximum element in a list",
    "Implement a function to sort a list using bubble sort",
    "Write a function to calculate the Fibonacci sequence",
    "Create a function to check if two strings are anagrams",
    "Implement a function to find the greatest common divisor",
    "Write a function to convert Celsius to Fahrenheit"
]

# Display example prompts in columns
cols = st.columns(2)
for i, example in enumerate(example_prompts):
    with cols[i % 2]:
        if st.button(f"Use: {example[:50]}...", key=f"example_{i}"):
            st.session_state.example_prompt = example
            st.rerun()

# Model information
st.header("Model Information")
st.info(f"""
**Model Type:** {model_type.upper()}
**Model Name:** {model_name}
**Device:** {model.device}
**Parameters:** {sum(p.numel() for p in model.model.parameters()):,}
""")

# Tips
st.header("Tips for Better Results")
st.markdown("""
- **Be specific**: Include details about the function name, parameters, and expected behavior
- **Use clear language**: Write prompts in simple, clear English
- **Include examples**: Mention specific use cases or examples when possible
- **Adjust parameters**: Experiment with temperature, top-p, and top-k values for different styles
- **Try different models**: Different models may perform better for different types of code
""")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and Transformers")

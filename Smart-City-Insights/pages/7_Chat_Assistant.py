
import streamlit as st
import random
import time

st.set_page_config(page_title="Chat Assistant", page_icon="💬")

st.title("💬 Sustainability Chat Assistant")

st.markdown("Ask me anything about urban sustainability, energy efficiency, or pollution control!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is the current AQI trend?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Simple rule-based response logic
    response = ""
    prompt_lower = prompt.lower()
    
    if "aqi" in prompt_lower or "air quality" in prompt_lower:
        response = "Based on current data, the AQI fluctuates between 100 and 200. It tends to peak during rush hours (8-10 AM, 5-7 PM) due to traffic."
    elif "energy" in prompt_lower:
        response = "The city's average energy consumption is 200 kWh per capita. Reducing this by 10% could improve our sustainability rating significantly."
    elif "traffic" in prompt_lower:
        response = "Traffic density is a major contributor to pollution. Our AI Vision module estimates current density is moderate."
    elif "green" in prompt_lower or "trees" in prompt_lower:
        response = "Green cover is currently at 30%. Increasing it to 40% is recommended by the AI Advisor."
    elif "hello" in prompt_lower or "hi" in prompt_lower:
        response = "Hello! I am your AI Smart City Assistant. How can I help you today?"
    else:
        response = "That's an interesting question. I'm currently monitoring AQI, Energy, and Traffic data. Try asking about those!"

    # Simulate typing delay
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for chunk in response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    
    # Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

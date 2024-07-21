# Wine Business Chatbot

## Introduction
This is a chatbot designed for a wine business website. It answers user queries based on a predefined corpus and sample Q&A. For any queries outside the corpus, it directs the user to contact the business directly.

## Features
- Minimalistic UI for user interaction.
- Context-aware conversation handling.
- Retrieval of relevant answers from a corpus.
- Uses OpenAI's GPT-4 for generating responses.

## Requirements
- Python 3.7+
- Streamlit
- OpenAI
- PyMuPDF
- SentenceTransformers
- Torch

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/wine-business-chatbot.git
    cd wine-business-chatbot
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Place your `corpus.pdf` and `saq.json` files in the root directory of the project.

5. Replace the placeholder with your actual OpenAI API key in `app.py`:
    ```python
    openai.api_key = 'your-api-key'
    ```

## Running the Chatbot
To run the chatbot, execute the following command:
```sh
streamlit run app.py

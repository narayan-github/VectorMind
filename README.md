# VectorMind

## Overview
VectorMind is a Streamlit-based chatbot that leverages Pydantic AI and Ollama for intelligent document retrieval and conversational AI. It integrates Supabase for database interactions and supports embeddings for improved search and response quality.

## Features
- **Conversational AI** powered by Ollama
- **Document Retrieval** using embedding-based search
- **Streamlit UI** for seamless interaction
- **Supabase Integration** for database storage
- **Retry Mechanism** with Tenacity for robustness

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.9+
- Virtual Environment (recommended)

### Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/vectormind.git
   cd vectormind
   ```

2. Create a virtual environment and activate it:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Create a `.env` file in the root directory.
   - Add the following variables:
     ```ini
     SUPABASE_URL=<your_supabase_url>
     SUPABASE_SERVICE_KEY=<your_supabase_service_key>
     OLLAMA_MODEL=llama3.2
     OLLAMA_BASE_URL=http://localhost:11434
     ```

## Usage
1. Start the Streamlit app:
   ```sh
   streamlit run app.py
   ```

2. Interact with the chatbot and ask questions about Pydantic AI.

## Project Structure
```
├── app.py                         # Main application file
├── pydantic_ai_documents.py       # Document processing module
├── site_pages.sql                 # SQL file for database setup
├── requirements.txt               # Dependencies
├── .env                           # Environment variables (not in repo)
├── .env.example                   # Example environment variables
├── README.md                      # Project documentation
```

## Deployment
To deploy the application:
- Use **Docker** or deploy via a cloud service like **Streamlit Sharing, Heroku, or AWS**.
- Ensure your Supabase and Ollama instances are correctly configured.

## Contributing
Feel free to contribute! Submit a PR with improvements or open an issue for bugs and enhancements.

## License
This project is licensed under the MIT License.

## Contact
For questions or support, reach out to Narayan Samal at [Email](narayancoding+VectorMind@gmail.com).


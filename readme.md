The main.py file uses mmr retrieving mechanism and main_compression_retriever.py uses contextual compression.

1. **Set Up a Virtual Environment (venv or conda)**  
   - Create and activate a new conda environment:  
     ```bash
     conda create --name langchain-env
     conda activate langchain-env
     ```
    - OR
    ```bash
     python -m venv langchain-env
     source langchain-env/bin/activate
     ```

2. **Create a `.env` File**  
   - Create a `.env` file in the project directory with the following content:  
     ```env
     HUGGINGFACE_ACCESS_TOKEN=your_huggingface_access_token
     PINECONE_API_KEY=your_pinecone_api_key
     ```
   - Replace `your_huggingface_access_token` and `your_pinecone_api_key` with your actual API keys.

3. **Install Libraries using `requirements.txt` File**

    **Using Conda:**
    ```bash
    conda install --file requirements.txt
    ```

    **Using Pip:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the `main.py` File**

    To execute the main script, use the following command:
    ```bash
    python main.py
    ```
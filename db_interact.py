import requests

# URL del server
url = 'http://localhost:5001'

# Funzione per salvare il vector store
def save_vector_store():
    response = requests.post(f'{url}/save_vector_store')
    return response.json()

# Funzione per caricare il vector store
def load_vector_store():
    response = requests.post(f'{url}/load_vector_store')
    return response.json()

# Esempio di utilizzo delle funzioni
if __name__ == '__main__':
    action = input("Do you want to save or load the vector store? (save/load): ").strip().lower()
    
    if action == 'save':
        save_response = save_vector_store()
        print("Save response:", save_response)
    elif action == 'load':
        load_response = load_vector_store()
        print("Load response:", load_response)
    else:
        print("Invalid action. Please enter 'save' or 'load'.")
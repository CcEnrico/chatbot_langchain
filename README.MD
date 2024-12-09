# Chatbot Tutorial

Benvenuti nel tutorial del progetto Chatbot. Questo progetto è stato creato per fornire un assistente virtuale che può rispondere a domande e interagire con gli utenti. Di seguito troverete una guida passo-passo su come configurare e utilizzare il chatbot.

## Prerequisiti

Assicurati di avere installato i seguenti strumenti:
- Python 3.12
- pip (Python package installer)

## Installazione

1. Clona il repository nella tua directory locale:
    ```bash
    git clone https://github.com/CcEnrico/chatbot_langchain
    ```
2. Naviga nella directory del progetto:
    ```bash
    cd chatbot
    ```
3. Installa le dipendenze richieste:
    ```bash
    pip install -r requirements.txt
    ```
4. Se preferisci, usa il comando seguente per creare un ambiente virtuale:
    ```bash
    python -m venv venv
    ```
    e attivalo con:
    ```bash
    source venv/bin/activate
    ```

## Configurazione

Prima di eseguire il chatbot, è necessario configurare alcuni parametri. Bisogna creare un file `.env` nella directory principale del progetto e inserire le variabili d'ambiente: 

## Docker

Se preferisci utilizzare Docker, puoi creare un'immagine e avviare un container con il seguente comando:
```bash
docker run -it -p 5001:5001 --name rag_server_container rag_server
```
oppure senza tenere traccia dello storico dei comandi:
```bash
docker run -p 5001:5001 --name rag_server_container rag_server
```

## Esecuzione

Per avviare il chatbot, esegui il seguente comando:
```bash
python main.py
```

Se non usi docker e vuoi avviare il server, esegui il seguente comando:
```bash
python src/RAG_server.py
```

## Utilizzo

Una volta avviato, il chatbot sarà pronto per interagire con te. Puoi iniziare a fare domande e il chatbot risponderà in base alla sua programmazione.

si puo anche dare file in imput Tramite RAG, usando:
```bash
python doc_loader.py
```

oppure se vuoi caricare un db da file o salvare in file 
```bash
python db_interact.py
```


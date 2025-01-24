
---

# Chatbot Tutorial

Benvenuti nel tutorial del progetto Chatbot. Questo progetto fornisce un assistente virtuale avanzato in grado di rispondere a domande, elaborare input e interagire con l'utente in modo dinamico. Utilizza un'architettura basata su Python e Docker per massimizzare flessibilità e facilità di utilizzo.

## Prerequisiti

Assicurati di avere installato i seguenti strumenti:
- Python 3.12 o versione successiva
- pip (Python package installer)
- Docker e Docker Compose (opzionale, per eseguire il progetto in container)

## Installazione

1. Clona il repository nella tua directory locale:
    ```bash
    git clone https://github.com/CcEnrico/chatbot_langchain
    ```
2. Naviga nella directory del progetto:
    ```bash
    cd chatbot
    ```
3. (Opzionale) Crea un ambiente virtuale per isolare le dipendenze:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
4. Installa le dipendenze richieste:
    ```bash
    pip install -r requirements.txt
    ```

## Configurazione

Crea un file `.env` nella directory principale del progetto per configurare le variabili d'ambiente necessarie. Ecco un esempio di `.env`:

```env
TOKENIZERS_PARALLELISM=false
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langchain_api_key
OPENAI_API_KEY=your_openai_api_key
```

Queste variabili sono essenziali per l'esecuzione corretta dei server LLM e RAG.

## Esecuzione con Docker

Per eseguire il progetto con Docker e Docker Compose:

1. Costruisci e avvia i container:
    ```bash
    docker-compose up --build
    ```
2. I servizi saranno accessibili:
   - LLM Server: `http://localhost:5002`
   - RAG Server: `http://localhost:5001`

## Note

### Configurazione dell'Endpoint

Quando si interagisce con il server RAG, è importante configurare correttamente l'endpoint a seconda dell'ambiente in cui si sta eseguendo il server(modificare file `llm_server.py`):

- Se stai eseguendo il server in locale:
    ```python
    response = requests.post(
        'http://localhost:5001/similarity_search',
        json={"query": [state["messages"][-1].content]}
    )
    ```

- Se stai eseguendo il server in un container Docker:
    ```python
    response = requests.post(
        'http://rag_server:5001/similarity_search',
        json={"query": [state["messages"][-1].content]}
    )
    ```

### Fermare e Riavviare i Container

- Per fermare i container:
    ```bash
    docker-compose down
    ```
- Per avviare tutti i container:
    ```bash
    docker-compose up
    ```
- Per riavviare tutti i container:
    ```bash
    docker-compose restart
    ```
- Per riavviare un container specifico (es. `llm_server`):
    ```bash
    docker-compose restart llm_server
    ```

## Accedere ai Container

Per accedere alla shell di un container (es. `llm_server`):
```bash
docker exec -it llm_server /bin/bash
```

Oppure, per il container `rag_server`:
```bash
docker exec -it rag_server /bin/bash
```

Una volta dentro il container, puoi esplorare il file system, eseguire script o eseguire comandi di debug.

## Esecuzione senza Docker

Se preferisci eseguire il progetto senza Docker:

1. Vai nella directory `src`:
    ```bash
    cd src
    ```
2. Avvia il server RAG:
    ```bash
    python RAG_server.py
    ```
3. In un altro terminale, vai nella directory `src` e avvia il server LLM:
    ```bash
    python LLM_server.py
    ```

## Utilizzo

### Interazione con il Chatbot
Una volta avviati i server, il chatbot sarà pronto per interagire. Puoi inviare domande e ottenere risposte in base alla logica programmata.

### Caricamento Documenti con RAG
Per aggiungere documenti al database RAG, utilizza il seguente comando dalla directory principale del progetto(bisogna mettere i documenti nella cartella `data` e modificare il file `doc_loader.py` e poi eseguire):
```bash
python doc_loader.py
```

### Interazione con il Database
Per salvare o caricare lo stato del database vettoria, utilizza il seguente comando dalla directory principale del progetto:
```bash
python db_interact.py
```

## Debug e Log

I log delle operazioni dei server vengono salvati nei rispettivi file di log (`server.log`) per facilitare il debug e il monitoraggio delle attività.

--- 
# Progetto ML

Il progetto si basa su modello basato sulle [CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network)
e in particolare sui loro utilizzi e applicazioni per l'analisi di dati audio. In particolare la
rete creata andrà ad analizzare il dataset [EMOVO](https://dagshub.com/kingabzpro/EMOVO), il quale
contiene registrazioni di discorsi parlati in lingua italiana, al cui interno vengono pronunciate
frasi con intonazioni e stati d'animo differenti da soggetti di sesso maschile e femminile.

## Requisiti di sistema

Il progetto è stato testato su sistemi con:

- Windows 11:
    - [WSL](https://learn.microsoft.com/en-us/windows/wsl/install)
    - CPU: Intel i7-8565u
    - GPU: Nvidia MX250 abilitata per utilizzo capacità di sviluppo CUDA
    - RAM: 16 GB

- MacOs Sonoma:
    - CPU: Chip Apple M2 8-core (4 performance, 4 efficiency)
    - GPU: 8-core
    - Neural Engine: 16-core
    - RAM: 8 GB

## Configurazione pacchetti python e dataset

Il progetto è impostato in modo tale da richiedere solamente l'esecuzione del file di configurazione
[`prepare.sh`](prepare.sh), al cui interno risiedono le istruzioni necessare per scaricare tutti i
pacchetti python utilizzati elencati dentro il file [`requirements.txt`](requirements.txt) e il
dataset di riferimento, decomprimerlo e renderlo utilizzabile, pertanto questo script dovrebbe
essere il primo passo da eseguire per un corretto funzionamento del progetto.

## Configurazione parametri modello

I parametri di configurazione, dentro il file [`base_config.yaml`](config/base_config.yaml) sono
adibiti alla configurazione degli iperparametri del modello, ovvero:

- **data**: parametri di configurazione del dataset

    | nome            | tipo  |  valori accettati  | descrizione                                                                      |
    | :-------------- | :---- | :----------------: | :------------------------------------------------------------------------------- |
    | **train_ratio** | float |      \[0, 1]       | proporzione di divisione del dataset in train e test                             |
    | **data_dir**    | str   | percorso directory | percorso directory dataset (precedentemente scaricati nella directory `dataset`) |

- **model**: parametri di configurazione del modello

    | nome        | tipo  | valori accettati | descrizione                                              |
    | :---------- | :---- | :--------------: | :------------------------------------------------------- |
    | **dropout** | float |     \[0, 1]      | percentuale dropout da applicare tra un layer e un altro |

- **training**: parametri di configurazione durante il training

    | nome                            | tipo  |                                       valori accettati                                        | descrizione                                                                                                                                            |
    | :------------------------------ | :---- | :-------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------- |
    | **epochs**                      | int   |                                                                                               | epoche per cui addestrare il modello                                                                                                                   |
    | **batch_size**                  | int   |                                                                                               | dimensione delle singole batch con cui addestrare il modello                                                                                           |
    | **optimizer**                   | str   | adadelta, adagrad, adamax, adamw, asgd, lbfgs, nadam, radam, rmsprop, rprop, sgd, sparse_adam | ottimizzatore da uilizzare                                                                                                                             |
    | **lr**                          | float |                                            \[0, 1]                                            | learning rate da uilizzare                                                                                                                             |
    | **warmup_ratio**                | float |                                            \[0, 1]                                            | percentuale di epoche dopo le quali il learning rate andrà a diminuire linearmente                                                                     |
    | **checkpoint_dir**              | str   |                                      percorso directory                                       | percorso directory in cui salvare il modello dopo la fine dell'addestramento                                                                           |
    | **model_name**                  | str   |                                                                                               | nome con cui salvare il modello dopo la fine dell'addestramento                                                                                        |
    | **device**                      | str   |                                        cpu, cuda, mps                                         | dispositivo di accelerazione hardware da utilizzare durante l'addestramento                                                                            |
    | **evaluation_metric**           | str   |                             accuracy, precision, recall, f1, loss                             | metrica da tenere in considerazione durante la valutazione del modello                                                                                 |
    | **best_metric_lower_is_better** | bool  |                                                                                               | indica se la metrica da tenere in considerazione durante la valutazione del modello è da considerarsi migliore se è inferiore o superiore a una soglia |

## Dataset

Il dataset si compone di tracce audio di frasi parlate in italiano da soggetti di sesso maschile e
femminile con diverse intonazioni ed emozioni. In particolare il dataset viene gestito dalle classi
dentro il file [`emovo_dataset.py`](data_classes/emovo_dataset.py).

Il dataset viene gestito dalla classe `EmovoDataset`, che ci permette di raccogliere, estrarre e
caricare i file audio grezzi.

### Inizializzazione

La classe richiede i seguenti parametri di inizializzazione:

| nome          | tipo | default | descrizione                                                                                                   |
| :------------ | :--- | :------ | :------------------------------------------------------------------------------------------------------------ |
| **data_path** | str  |         | il precorso della directory in cui cercare i file audio (precedentemente scaricati nella directory `dataset`) |
| **train**     | bool | True    | se il modello verrà utilizzato durante fase di addestramento o durante la fase di test                        |
| **resample**  | bool | True    | se applicare il resamplig a 16Khz alle tracce audio                                                           |

successivamente eseguirà i seguenti passaggi:

- scansiona le directory contenenti i file audio, e per ogni file audio:
    - estrae le [features](https://dagshub.com/kingabzpro/EMOVO#organization-of-the-dataset) rilevanti
        dal percorso del file audio (labels)
    - carica il file audio
    - registra il file audio di lunghezza maggiore
    - immagazzina il percorso del file audio e le sue features rilevanti
- aggiusta il valore della lunghezza del file audio più lungo per tener conto dell'aggiunta di
    padding alle tracce di lunghezza inferiore alla lunghezza massima precedentemente calcolata se
    necessario
- applica un resampling a 16Khz al valore della lunghezza del file audio più lungo se richiesto
    dall'utente

### Estrazione del file audio grezzo

L'estrazione dei file audio e le informazion ad esso relative verranno estratte in maniera lazy
dentro il metodo `__getitem__`, che prendendo come input l'indice del file audio da estrarre,
andrà a:

- recuperare il percorso del file e la label corrispondente
- caricare la waveform e il sample rate associato
- applicare il resampling a 16Khz se richiesto precedentemente e se necessario
- controllare che la traccia sia in formato stereo ed eventualmente duplicare il canale mono nel
    caso contrario
- applicare il padding, sottoforma di audio vuoto (silenzio), alle tracce audio che lo richiedono
    per uniformarsi alla lunghezza massima precedentemente calcolata

Infine ritornerà un'istanza della classe `Sample`, ovvero un dizionario contenente i seguenti campi:

| nome            | tipo         | descrizione                                                                                                                  |
| :-------------- | :----------- | :--------------------------------------------------------------------------------------------------------------------------- |
| **waveform**    | torch.Tensor | il tensore che ci descrive in maniera grezza la forma d'onda del file audio                                                  |
| **sample_rate** | int          | il sample rate della forma d'onda del file audio grezzo                                                                      |
| **label**       | int          | la label relativa alla [feature]((https://dagshub.com/kingabzpro/EMOVO#organization-of-the-dataset)) associata al file audio |

## Modello CNN

Il modello che andrà ad analizzare e ad imparare il dataset è basato su un'architettura CNN,
implementata con la classe `EmovoCNN` secondo l'architettura descritta in
[`cnn_model.py`](model_classes/cnn_model.py).

### Inizializzazione

La classe richiede i seguenti parametri di inizializzazione:

| nome              | tipo         | valori accettati                                               | descrizione                                                                                   |
| :---------------- | :----------- | :------------------------------------------------------------- | :-------------------------------------------------------------------------------------------- |
| **waveform_size** | int          |                                                                | la dimensione della waveform che andrà analizzata dal modello, ovvero l'input al primo strato |
| **dropout**       | float        | \[0, 1]                                                        | percentuale dropout da applicare tra un layer e un altro                                      |
| **device**        | torch.device | torch.device("cpu"), torch.device("cuda"), torch.device("mps") | dispositivo di accelerazione hardware da utilizzare durante l'addestramento                   |

### Addestramento e test

L'addestramento inizia leggendo i [parametri di configurazione](#configurazione-parametri-modello)
precedenetemente descritti, quindi eseguendo i passaggi in [`train.py`](train.py).

Dopo l'addestramento il modello risultante verrà salvato nella **checkpoint_dir** specificata,
pertanto sarà possibile valutarne e testarne le prestazioni successivamente mediante i passaggi in
[`test.py`](test.py), con i criteri e i metodi di valutazione descritti in [`metrics.py`](metrics.py).

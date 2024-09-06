## Execução

Para executar o programa, abra o terminal e navegue até a pasta raíz do projeto e em seguida, execute o seguinte comando:

```
python -m venv venv
source venv/bin/activate  # Para Linux/MacOS
venv\Scripts\activate  # Para Windows
```

Após isso, dentro da pasta /src instale as dependências usando o seguinte comando: 

```
pip install -r requirements.txt
```

Para executar os experimentos, ainda na pasta /src execute para:

Análise da eficácia dos algoritmos GA e DE:

```
streamlit run experimento_1.py
```
Experimento para o problema de restrição redutor de velocidade:

```
streamlit run experimento_2.py
```

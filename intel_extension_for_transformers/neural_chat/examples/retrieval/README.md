This example demonstrates how to enhance the generation ability of your chatbot with your local documents within few lines of codes.

# Introduction
Large Language Models (LLMs) have demonstrated remarkable performance in various Natural Language Processing (NLP) tasks. Compared to earlier pretrained models, LLMs can produce better results on tasks without fine-tuning, reducing the cost of use. The popularity of applications like ChatGPT has attracted many users seeking to address everyday problems. However, some users have encountered a challenge known as "model hallucination," where LLMs generate incorrect or nonexistent information, raising concerns about content accuracy.

To improve the accuracy of generated content, two approaches can be considered: expanding the training data or utilizing an external database. Expanding the training data is impractical due to the time and effort required to train a high-performance LLM. It's challenging to collect and maintain an extensive, up-to-date knowledge corpus. Therefore, we propose an economically efficient alternative: leveraging relevant documents from a local database during content generation. These retrieved documents will be integrated into the input prompt of the LLM to enhance the accuracy and reliability of the generated results.

Before deploying this example, please follow the instructions in the [README](../../README.md) to install the necessary dependencies.

# Usage
The Neural Chat API offers an easy way to create and utilize chatbot models while integrating local documents. Our API simplifies the process of automatically handling and storing local documents in a document store. We provide support for two retrieval methods:
1. Dense Retrieval: This method is based on document embeddings, enhancing the accuracy of retrieval. Learn more about [here](https://medium.com/@aikho/deep-learning-in-information-retrieval-part-ii-dense-retrieval-1f9fecb47de9) (based on the document embedding) 
2. Sparse Retrieval: Using TF-IDF, this method efficiently retrieves relevant information. Explore this approach in detail [here](https://medium.com/itnext/deep-learning-in-information-retrieval-part-i-introduction-and-sparse-retrieval-12de0423a0b9).

## Import the module and set the retrieval config:
The user can download the [Intel 2022 Annual Report](https://d1io3yog0oux5.cloudfront.net/_897efe2d574a132883f198f2b119aa39/intel/db/888/8941/file/412439%281%29_12_Intel_AR_WR.pdf) for a test.

```python
from intel_extension_for_transformers.neural_chat import PipelineConfig
from intel_extension_for_transformers.neural_chat import plugins
plugins.retrieval.enable=True
plugins.retrieval.args["input_path"]="./Annual_report.pdf"
config = PipelineConfig(plugins=plugins)
```

## Build the chatbot and interact with the chatbot:

```python
from intel_extension_for_transformers.neural_chat import build_chatbot
chatbot = build_chatbot(config)
response = chatbot.predict("What is IDM 2.0?")
```

## Deploying through shell comand:
```shell
python retrieval_chat.py
```

# Parameters
The user can costomize the retrieval parameters to meet the personal demmads for better catering the local files. The user can set the specific parameter by plugins.retrieval.args["xxx"]. Below the description of each available parameters,
```
>>>persist_dir [str]: The local path to save the processed database. Default to "./output".

>>>process [bool]: Select to process the too long document into small chucks. Default to "True".

>>>input_path [str]: The user local path to a file folder or a specific file path. The code itself will check the path is a folder or a file. If it is a folder, the code will process all the files in the given folder. If it is a file, the code will prcess this single file.

>>>embedding_model [str]: the user specific document embedding model for dense retrieval. The user could selecte a specific embedding model from "https://huggingface.co/spaces/mteb/leaderboard". Default to "hkunlp/instructor-large". 

>>>max_length [int]: The max context length in the processed chucks. Should be combined with "process". Default to "512".

>>>retrieval_type [str]: Select a method for retrieval from "dense" or "sparse". Default to "dense".

>>>document_store [str]: Considering the sparse retrieval needs to load the data into memory. We provide "InMemoryDocumentStore" and "ElasticsearchDocumentStore" for manage the memory efficiency for sparse retrieval. 
    
>>>top_k [int]: The number of the retrieved documents. Default to "1".

>>>search_type [str]: Select a ranking method for dense retrieval from "mmr", "similarity" and "similarity_score_threshold". "similarity" will return the most similar docs to the input query. "mmr" will rank the docs using the maximal marginal relevance method. "similarity_score_threshold" will return the mosy similar docs that also meet the threshold. Deault to "mmr".

>>>search_kwargs [dict]: Used by dense retrieval. Should be in the same format like {"k":1, "fetch_k":5}. "k" is the amount of documents to return. "score_threshold" is the minimal relevance threshold for "similarity_score_threshold" search. "lambda_mult" is the diversity of results returned by "mmr". "fetch_k" determines the amount of documents to pass to the "mmr" algorithm. Default to {"k":1, "fetch_k":5}.
```


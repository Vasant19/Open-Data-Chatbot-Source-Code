# Open-Data-Chatbot-Source-Code-respository

## Overview
The Open Data Chatbot is an intelligent conversational agent designed to assist users in interacting with open data through natural language queries. This project allows users to retrieve insights from various datasets easily. 

## Project Context
We developed this as a proof of concept for an internal chatbot for IRCC (Immigration, Refugees and Citizenship Canada), equivalent to the chatbot available at the National Institutes of Health(https://irp.nih.gov/catalyst/32/2/news-you-can-use-nhlbi-chat), but with enhanced accuracy. This chatbot serves as a single source of truth, reducing the risk of hallucinations and providing reliable information.

## Demo
#### Current Developed State 
https://github.com/user-attachments/assets/2a68702a-8c66-4d33-91aa-e576d7c5a428

#### Further Production environment steps Integrating using Oauth 2.0, Nextjs, TailwindCSS and Django.
https://github.com/user-attachments/assets/2a2de702-0f9c-4665-90e8-4cd0e71a247e


## Setup
```pip install -r requirements.txt```

```git clone "<my-repo-url>"```

```streamlit run streamlithome.py  #For main Branch ```

```npm run dev #For master branch```

## Repo Architecture 
This project utilizes a **monorepo architecture**, which consolidates multiple related projects and components into a single repository.
In this monorepo, each component of the chatbot is organized as a single script file, making it easy to test, modify, and execute specific functionalities independently. The structure supports clear separation of different functionalities while maintaining a cohesive environment for development. Additionally, the scripts are written without encapsulation to maintain simplicity, which enhances readability. Brief comments are included to provide context and further improve understanding.


## Branches
This repository contains the following branches:

**Main**: The Primary branch for intial development and Contains the final output, with stable version. Follows File-based routing

**Master**: The Second version of the project containing Futher production steps.

**PandasAI**: A general-purpose everyday chatbot utilizing BambooLLM. Please note that this branch does not include advanced features such as prompt engineering, RAG, database storage, agent training, or vector similarity search, which may result in less accurate outputs. A sample visualization file is also present in this branch for basic visualizations without requiring external frameworks.

## Tools Used
The Open Data Chatbot project utilized various tools across different stages of its development. The toolkit evolved as we explored different technologies to best suit our needs(Highlighted words signify tech stack used for final output). Below is the final set of tools and technologies chosen:

![Toolkit used](https://github.com/user-attachments/assets/350f452d-3ba2-4c3d-8019-9203f50d29f1)
Each tool was selected after experimenting with different alternatives to ensure the best fit for the project’s requirements.

## Maintenance
Please note that I will not be maintaining the dependencies of this project. Additionally, there may be some placeholders or previous commits that could disrupt the code's execution. If you have any questions or feedback, feel free to reach out via email or create a pull request.

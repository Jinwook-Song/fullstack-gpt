# Fullstack GPT

랭체인으로 AI 웹 서비스 7개 만들기

| 프로젝트 기간 | 23.11.11 ~                                                                                                                              |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| 프로젝트 목적 | Langchain, Language Models 에 대한 기본 이해, 자체 데이터에 GPT-4를 사용하는 방법, 커스텀 자율 에이전트(Autonomous Agent)를 만드는 방법 |
| Github        | https://github.com/Jinwook-Song/fullstack-gpt                                                                                           |

## TODOS

- AI 웹 서비스 (6종) : DocumentGPT, PrivateGPT, QuizGPT, SiteGPT, MeetingGPT, InvestorGPT
- ChatGPT 플러그인 (1종) : ChefGPT
- 활용하는 패키지 : Langchain, GPT-4, Whisper, FastAPI, Streamlit, Pinecone, Hugging Face, ...

### DocumentGPT

법률. 의학 등 어려운 용어로 가득한 각종 문서. AI로 빠르게 파악하고 싶다면?

AI로 신속하고 정확하게 문서 내용을 파악하고 정리한 뒤, 필요한 부분만 쏙쏙 골라내어 사용하세요. DocumentGPT 챗봇을 사용하면, AI가 문서(.txt, .pdf, .docx 등)를 꼼꼼하게 읽고, 해당 문서에 관한 질문에 척척 답변해 줍니다.

### PrivateGPT

회사 기밀이 유출될까 걱정된다면? 이제 나만이 볼 수 있는 비공개 GPT를 만들어 활용하세요!

DocumentGPT와 비슷하지만 로컬 언어 모델을 사용해 비공개 데이터를 다루기에 적합한 챗봇입니다. 데이터는 컴퓨터에 보관되므로 오프라인에서도 사용할 수 있습니다. 유출 걱정 없이 필요한 데이터를 PrivateGPT에 맡기고 업무 생산성을 높일 수 있어요.

### QuizGPT

암기해야 할 내용을 효율적으로 학습하고 싶다면?

문서나 위키피디아 등 학습이 필요한 컨텐츠를 AI에게 학습시키면, 이를 기반으로 퀴즈를 생성해 주는 앱입니다. 번거로운 과정을 최소화하고 학습 효율을 극대화할 수 있어, 특히 시험이나 단기간 고효율 학습이 필요할 때 매우 유용하게 사용할 수 있어요.

### SiteGPT

자주 묻는 질문 때문에 CS 직원을 채용...? SiteGPT로 비용을 2배 절감해 봅시다.

웹사이트를 스크랩하여 콘텐츠를 수집하고, 해당 출처를 인용하여 관련 질문에 답변하는 챗봇입니다. 고객 응대의 대부분을 차지하는 단순 정보 안내에 들이는 시간을 획기적으로 줄일 수 있고, 고객 또한 CS직원의 근무 시간에 구애받지 않고 정확한 정보를 빠르게 전달받을 수 있습니다.

### MeetingGPT

이제 회의록 정리는 MeetingGPT에게 맡기세요!

회의 영상 내용을 토대로 오디오 추출, 콘텐츠를 수집하여 회의록을 요약 및 작성해 주는 앱입니다. 회의 내용을 기록하느라 회의에 제대로 참석하지 못하는 일을 방지할 수 있고, 관련 질의응답도 가능해 단순한 기록보다 훨씬 더 효율적으로 회의록을 관리하고 활용할 수 있습니다.

### InvestorGPT

AI가 자료 조사도 알아서 척척 해 줍니다.

인터넷을 검색하고 타사 API를 사용할 수 있는 자율 에이전트입니다. 회사, 주가 및 재무제표를 조사하여 재무에 대한 인사이트를 제공할 수 있습니다. 또한 알아서 데이터베이스를 수집하기 때문에 직접 SQL 쿼리를 작성할 필요가 없고, 해당 내용에 대한 질의응답도 얼마든지 가능합니다.

### ChefGPT

요즘 핫한 ChatGPT 플러그인? 직접 구현해 봐요!

유저가 ChatGPT 플러그인 스토어에서 설치할 수 있는 ChatGPT 플러그인입니다. 이 플러그인을 통해 유저는 ChatGPT 인터페이스에서 바로 레시피를 검색하고 조리법을 얻을 수 있습니다. 또한 ChatGPT 플러그인에서 OAuth 인증을 구현하는 방법에 대해서도 배웁니다.

- prompt & template

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate

chat = ChatOpenAI(temperature=0.1)  # randomness of response

template = PromptTemplate.from_template(
    "What is the distance between {country_a} and {country_b}",
)

prompt = template.format(country_a="Mexico", country_b="Korea")

chat.predict(prompt)

template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a geography expert. And you only reply in {language}."),
        ("ai", "Ciao, mi chiamo {name}!"),
        (
            "human",
            "What is the distance between {country_a} and {country_b}. Also, what is your name?",
        ),
    ]
)

prompt = template.format_messages(
    language="Greek", name="Socrates", country_a="Mexico", country_b="Korea"
)

chat.predict_messages(prompt)
```

## LangChain

- prompt & template

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate

chat = ChatOpenAI(temperature=0.1)  # randomness of response

template = PromptTemplate.from_template(
    "What is the distance between {country_a} and {country_b}",
)

prompt = template.format(
    country_a="Mexico",
    country_b="Korea",
)

chat.predict(prompt)
```

```python
template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a geography expert. And you only reply in {language}.",
        ),
        (
            "ai",
            "Ciao, mi chiamo {name}!",
        ),
        (
            "human",
            "What is the distance between {country_a} and {country_b}. Also, what is your name?",
        ),
    ]
)

prompt = template.format_messages(
    language="Greek", name="Socrates", country_a="Mexico", country_b="Korea"
)

chat.predict_messages(prompt)
```

- parser

```python
from langchain.schema import BaseOutputParser

class CommaOutputParser(BaseOutputParser):
    def parse(self, text):
        items = text.strip().split(",")
        return list(map(str.strip, items))
```

```python
template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a list geterating machine.Everything you are asked will be answered with a comma seperated list of max {max_items} in lowercase. Do NOT reply with anything else.",
        ),
        (
            "human",
            "{question}",
        ),
    ]
)

prompt = template.format_messages(max_items=10, question="What are the planets?")

result = chat.predict_messages(prompt)

p = CommaOutputParser()

p.parse(result.content)
```

- chain [docs](https://python.langchain.com/docs/expression_language/interface)
  - 위의 과정을 chain으로 단순화 할 수 있다.

```python
chain = template | chat | CommaOutputParser()

chain.invoke({"max_items": 5, "question": "What are the pokemons?"})
```

- chaing chains

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate

chat = ChatOpenAI(temperature=0.1)  # randomness of response

chef_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a world-calss international chef. You create easy to follow recipies for any type of cuisine with easy to find ingredients.",
        ),
        ("human", "I want to cook {cuisine} food."),
    ]
)

chef_chain = chef_prompt | chat

veg_chef_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a vegetarian chef specialized on making traditional recipies vegetarian. You find alternative ingredients and explain their preparation. You don't radically modify the recipe. If there is no alternative for a food just say you don't know how to replace it.",
        ),
        ("human", "{recipe}"),
    ]
)

veg_chain = veg_chef_prompt | chat

final_chain = {"recipe": chef_chain} | veg_chain

final_chain.invoke({"cuisine": "indian"})
```

## Model IO ([docs](https://python.langchain.com/docs/modules/))

- \***\*FewShotPromptTemplate\*\***
  - 특정 형식으로 답변할 수 있도록 해줌

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler

chat = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

examples = [
    {
        "question": "What do you know about France?",
        "answer": """
        Here is what I know:
        Capital: Paris
        Language: French
        Food: Wine and Cheese
        Currency: Euro
        """,
    },
    {
        "question": "What do you know about Italy?",
        "answer": """
        I know this:
        Capital: Rome
        Language: Italian
        Food: Pizza and Pasta
        Currency: Euro
        """,
    },
    {
        "question": "What do you know about Greece?",
        "answer": """
        I know this:
        Capital: Athens
        Language: Greek
        Food: Souvlaki and Feta Cheese
        Currency: Euro
        """,
    },
]

example_template = """
    Human: {question}
    AI: {answer}
"""

example_prompt = PromptTemplate.from_template(example_template)

prompt = FewShotPromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
    suffix="Human: What do you know about {country}?",
    input_variables=["country"],
)

chain = prompt | chat

chain.invoke({"country": "Germany"})
```

- FewShotChatMessagePromptTemplate

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler

chat = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

examples = [
    {
        "country": "France",
        "answer": """
        Here is what I know:
        Capital: Paris
        Language: French
        Food: Wine and Cheese
        Currency: Euro
        """,
    },
    {
        "country": "Italy",
        "answer": """
        I know this:
        Capital: Rome
        Language: Italian
        Food: Pizza and Pasta
        Currency: Euro
        """,
    },
    {
        "country": "Greece",
        "answer": """
        I know this:
        Capital: Athens
        Language: Greek
        Food: Souvlaki and Feta Cheese
        Currency: Euro
        """,
    },
]

# only to format examples
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "What do you know about {country}?"),
        ("ai", "{answer}"),
    ]
)

example_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a geography expert, you give short answers."),
        example_prompt,
        ("human", "What do you know about {country}?"),
    ]
)

chain = final_prompt | chat

chain.invoke({"country": "Korea"})
```

- \***\*LengthBasedExampleSelector\*\***

```python
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.prompts.few_shot import (
    FewShotPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain.prompts.example_selector import LengthBasedExampleSelector

chat = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

examples = [
    {
        "question": "What do you know about France?",
        "answer": """
        Here is what I know:
        Capital: Paris
        Language: French
        Food: Wine and Cheese
        Currency: Euro
        """,
    },
    {
        "question": "What do you know about Italy?",
        "answer": """
        I know this:
        Capital: Rome
        Language: Italian
        Food: Pizza and Pasta
        Currency: Euro
        """,
    },
    {
        "question": "What do you know about Greece?",
        "answer": """
        I know this:
        Capital: Athens
        Language: Greek
        Food: Souvlaki and Feta Cheese
        Currency: Euro
        """,
    },
]

example_prompt = PromptTemplate.from_template("Human: {question}\nAI:{answer}")

example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=180,
)

prompt = FewShotPromptTemplate(
    example_prompt=example_prompt,
    example_selector=example_selector,
    suffix="Human: What do you know about {country}?",
    input_variables=["country"],
)

prompt.format(country="Brazil")
```

- Custom Selector
  - selector를 구현할 수 있다.
  - `add_example`, `select_examples` method를 구현해야 한다

```python
class RandomExampleSelector(BaseExampleSelector):
    def __init__(self, examples):
        self.examples = examples

    def add_example(self, example):
        self.examples.append(example)

    def select_examples(self, input_variables):
        from random import choice

        return [choice(self.examples)]

example_prompt = PromptTemplate.from_template("Human: {question}\nAI:{answer}")

example_selector = RandomExampleSelector(examples=examples)

prompt = FewShotPromptTemplate(
    example_prompt=example_prompt,
    example_selector=example_selector,
    suffix="Human: What do you know about {country}?",
    input_variables=["country"],
)

prompt.format(country="Brazil")
```

- Serialization with PipelinePromptTemplate

```python
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.prompts.pipeline import PipelinePromptTemplate

chat = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

intro = PromptTemplate.from_template(
    """
    You are a role playing assistant.
    And you are impersonating a {character}
    """
)

example = PromptTemplate.from_template(
    """
    This is an example of how you talk:
    Human: {example_question}
    You: {example_answer}
    """
)

start = PromptTemplate.from_template(
    """
    Start now!

    Human: {question}
    You:
    """
)

final = PromptTemplate.from_template(
    """
    {intro}

    {example}

    {start}
    """
)

pipeline_prompts = [
    ("intro", intro),
    ("example", example),
    ("start", start),
]

full_prompt = PipelinePromptTemplate(
    pipeline_prompts=pipeline_prompts,
    final_prompt=final,
)

chain = full_prompt | chat

chain.invoke(
    {
        "character": "Pirate",
        "example_question": "What is your location?",
        "example_answer": "Arrrrg! That is a secret!! Arg arg!",
        "question": "What is your favorite food?",
    }
)
```

- caching

```python
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.globals import set_llm_cache, set_debug
from langchain.cache import InMemoryCache, SQLiteCache

set_llm_cache(SQLiteCache("cache.db"))
set_debug(True)

chat = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

chat.predict("How do you make italian pasta")
```

## Memory ([docs](https://python.langchain.com/docs/modules/memory/))

- ConversationBufferMemory

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(return_messages=True)

memory.save_context(
    {"input": "Hi"},
    {"output": "How are you?"},
)

memory.load_memory_variables({})
```

- ConversationBufferWindowMemory
  저장할 범위 지정

```python
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(
    return_messages=True,
    k=4,
)
```

- ConversationSummaryMemory

```python
from langchain.memory import ConversationSummaryMemory
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0.1)

memory = ConversationSummaryMemory(llm=llm)

def add_message(input, output):
    memory.save_context({"input": input}, {"output": output})

def get_history():
    return memory.load_memory_variables({})

add_message(
    "Hi I'm Jinwook, I live in South Korea",
    "Wow that is so cool!",
)
```

- ConversationSummaryBufferMemory
  token limit 이전 message에 대해서 summary

```python
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0.1)

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=100,
    return_messages=True,
)

def add_message(input, output):
    memory.save_context({"input": input}, {"output": output})

def get_history():
    return memory.load_memory_variables({})

add_message("Hi I'm Jinwook, I live in South Korea", "Wow that is so cool!")
```

- ConversationKGMemory

```python
from langchain.memory import ConversationKGMemory
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0.1)

memory = ConversationKGMemory(
    llm=llm,
    return_messages=True,
)

def add_message(input, output):
    memory.save_context({"input": input}, {"output": output})

add_message("Hi I'm Jinwook, I live in South Korea", "Wow that is so cool!")
```

- \***\*Memory on LLMChain\*\***
  chat history에 대한 정보를 넣어준다

```python
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(temperature=0.1)

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=120,
    memory_key="chat_history",
)

template = """
    You are a helpful AI talking to a human.

    {chat_history}
    Human:{question}
    You:
"""

chain = LLMChain(
    llm=llm,
    memory=memory,
    prompt=PromptTemplate.from_template(template),
    verbose=True,
)

chain.predict(question="My name is jinwook")
```

- Chat based memory

```python
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

llm = ChatOpenAI(temperature=0.1)

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=120,
    memory_key="chat_history",
    return_messages=True,
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI talking to a human"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

chain = LLMChain(
    llm=llm,
    memory=memory,
    prompt=prompt,
    verbose=True,
)

chain.predict(question="My name is jinwook")
```

- \***\*LCEL Based Memory\*\***

```python
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

llm = ChatOpenAI(temperature=0.1)

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=120,
    memory_key="chat_history",
    return_messages=True,
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI talking to a human"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

def load_memory(_):
    return memory.load_memory_variables({})["chat_history"]

def invoke_chain(question):
    result = chain.invoke({"question": question})
    memory.save_context(
        {"input": question},
        {"output": result.content},
    )
    print(result)

chain = RunnablePassthrough.assign(chat_history=load_memory) | prompt | llm

invoke_chain("My name is jinwook")
```

## RAG (Retrieval-Augmented Generation)

```
"RAG"는 "Retrieval-Augmented Generation"의 약자로, "검색-증강 생성"이라는 의미를 가지고 있습니다. 이는 자연어 처리(NLP) 및 기계 학습 분야, 특히 챗봇이나 질문-응답 시스템과 같은 고급 언어 모델을 구축하는 데 사용되는 기술입니다.

RAG에 대한 간략한 개요는 다음과 같습니다:

검색과 생성의 결합: RAG는 NLP의 두 가지 주요 구성 요소인 정보 검색과 응답 생성을 결합합니다. 검색 부분은 관련 정보를 찾기 위해 대규모 데이터베이스나 문서 컬렉션을 검색하는 과정을 포함합니다. 생성 부분은 검색된 정보를 바탕으로 일관되고 맥락적으로 적절한 텍스트를 생성하는 과정입니다.
작동 방식: RAG 시스템에서 질문이나 프롬프트가 주어지면 모델은 먼저 질문에 대한 답변을 제공하는 데 유용한 정보를 포함할 수 있는 관련 문서나 텍스트를 검색합니다. 그런 다음 이 검색된 정보를 생성 모델에 공급하여 일관된 응답을 합성합니다.
장점: RAG의 주요 장점은 모델이 외부 지식을 활용할 수 있게 하여 보다 정확하고 상세하며 맥락적으로 관련된 답변을 제공할 수 있다는 것입니다. 이는 특정 지식이나 사실적 정보가 필요한 질문에 특히 유용합니다.
응용 분야: RAG는 챗봇, 질문-응답 시스템 및 정확하고 맥락적으로 관련된 정보를 제공하는 것이 중요한 다른 AI 도구와 같은 다양한 응용 분야에 사용됩니다. 특히 모델이 다양한 주제와 데이터를 기반으로 이해하고 응답을 생성해야 하는 상황에서 유용합니다.
개발 및 사용: AI 및 기계 학습 커뮤니티에서 RAG는 다양한 연구 논문과 구현이 개발되고 있으며 주요 초점 중 하나입니다. 이는 학습된 정보뿐만 아니라 외부 소스에서 새롭고 관련된 정보를 통합하여 응답의 질과 관련성을 향상시키는 더 정교한 AI 시스템으로 나아가는 단계를 나타냅니다.
```

[docs](https://python.langchain.com/docs/modules/chains/document/)

- load and split document
  - UnstructuredFileLoader 임의의 포맷 문서를 load 할 수 있다
  - RecursiveCharacterTextSplitter
  - CharacterTextSplitter (seperator를 지정할 수 있다)
    - chunk_overlap 이전의 text를 어느정도로 겹치게 나눌지
    - from_tiktoken_encoder: token을 기반으로 모델이 텍스트를 세는것과 동일하게
  ```python
  from langchain.document_loaders import UnstructuredFileLoader
  from langchain.text_splitter import CharacterTextSplitter

  text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
      separator="\n",
      chunk_size=600,
      chunk_overlap=100,
  )

  loader = UnstructuredFileLoader("./files/chapter_one.docx")

  docs = loader.load_and_split(text_splitter=text_splitter)

  len(docs)
  ```
- Embedding
  text를 vector로 변환

```python
from langchain.embeddings import OpenAIEmbeddings

embedder = OpenAIEmbeddings()

# get vertor for the text
vector = embedder.embed_documents(texts=["hello", "how", "are", "you"])

vector
```

```python
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator="\n",
    chunk_size=600,
    chunk_overlap=100,
)

loader = UnstructuredFileLoader("./files/chapter_one.docx")

documents = loader.load_and_split(text_splitter=text_splitter)

vectorstore = Chroma.from_documents(documents=documents, embedding=OpenAIEmbeddings())
```

- RetrievalQA
  `stuff`
  : This chain takes a list of documents and formats them all into a prompt, then passes that prompt to an LLM. It passes ALL documents, so you should make sure it fits within the context window the LLM you are using.
  `refine`
  : This chain collapses documents by generating an initial answer based on the first document and then looping over the remaining documents to *refine* its answer. This operates sequentially, so it cannot be parallelized. It is useful in similar situatations as MapReduceDocuments Chain, but for cases where you want to build up an answer by refining the previous answer (rather than parallelizing calls).
  `map-reduce`
  : This chain first passes each document through an LLM, then reduces them using the ReduceDocumentsChain. Useful in the same situations as ReduceDocumentsChain, but does an initial LLM call before trying to reduce the documents.
  `map-rerank`
  : This calls on LLM on each document, asking it to not only answer but also produce a score of how confident it is. The answer with the highest confidence is then returned. This is useful when you have a lot of documents, but only want to answer based on a single document, rather than trying to combine answers (like Refine and Reduce methods do).

```python
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.vectorstores.faiss import FAISS
from langchain.chains import RetrievalQA
from langchain.storage import LocalFileStore

cache_dir = LocalFileStore("./.cache/")

llm = ChatOpenAI()

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator="\n",
    chunk_size=600,
    chunk_overlap=100,
)

loader = UnstructuredFileLoader("./files/chapter_one.docx")

documents = loader.load_and_split(text_splitter=text_splitter)

embeddings = OpenAIEmbeddings()

cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    embeddings,
    cache_dir,
)

vectorstore = FAISS.from_documents(
    documents=documents,
    embedding=cached_embeddings,
)

chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
)

# chain.run("Where does Winston live?")
chain.run("Describe Victory Mansions")
```

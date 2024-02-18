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

- 다양한 포맷의 문서 load → split (비용 절감 및 LLM의 성능을 위해) → embedding (text를 의미별로 벡터화 + caching) → RetrivalQA chain 생성 (다양한 chain type)

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

- Stuff LCEL Chain
  | Component | Input Type | Output Type |
  | ------------ | ----------------------------------------------------- | --------------------- |
  | Prompt | Dictionary | PromptValue |
  | ChatModel | Single string, list of chat messages or a PromptValue | ChatMessage |
  | LLM | Single string, list of chat messages or a PromptValue | String |
  | OutputParser | The output of an LLM or ChatModel | Depends on the parser |
  | Retriever | Single string | List of Documents |
  | Tool | Single string or dictionary, depending on the tool | Depends on the tool |
  예시에서 `Describe Victory Mansions`를 input으로 받아 Document list를 반환

  ```python
  from langchain.chat_models import ChatOpenAI
  from langchain.storage import LocalFileStore
  from langchain.document_loaders import UnstructuredFileLoader
  from langchain.text_splitter import CharacterTextSplitter
  from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
  from langchain.vectorstores.faiss import FAISS
  from langchain.prompts import ChatPromptTemplate
  from langchain.schema.runnable import RunnablePassthrough

  cache_dir = LocalFileStore("./.cache/")

  llm = ChatOpenAI(temperature=0.1)

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

  retriver = vectorstore.as_retriever()

  prompt = ChatPromptTemplate.from_messages(
      [
          (
              "system",
              """
  You are a helpful assistant. Answer questions using only the context. If you don't know the answer, just say you don't know, don't make it up:\n\n{context}
  """,
          ),
          ("human", "{question}"),
      ]
  )

  chain = {"context": retriver, "question": RunnablePassthrough()} | prompt | llm

  chain.invoke("Describe Victory Mansions")
  ```

- \***\*Map Reduce LCEL Chain\*\***

  ```python
  map_doc_prompt = ChatPromptTemplate.from_messages(
      [
          (
              "system",
              """
              Use the following portion of a long document to see if any of the text is relevant to answer the question. Return any relevant text verbatim. If there is no relevant text, return : ''
              -------
              {context}
              """,
          ),
          ("human", "{question}"),
      ]
  )

  map_doc_chain = map_doc_prompt | llm

  def map_docs(inputs):
      documents = inputs["documents"]
      question = inputs["question"]
      results = []
      for document in documents:
          result = map_doc_chain.invoke(
              {"context": document.page_content, "question": question}
          ).content
          results.append(result)
      results = "\n\n".join(results)
      return results

  map_chain = {
      "documents": retriever,
      "question": RunnablePassthrough(),
  } | RunnableLambda(map_docs)

  final_prompt = ChatPromptTemplate.from_messages(
      [
          (
              "system",
              """
  Given the following extracted parts of a long document and a question, create a final answer.
  If you don't know the answer, just say that you don't know. Don't try to make up an answer.
  -----
  {context}
  """,
          ),
          ("human", "{question}"),
      ]
  )

  chain = {"context": map_chain, "question": RunnablePassthrough()} | final_prompt | llm

  # chain.invoke("Describe Victory Mansions")
  chain.invoke("Where dos Winston go to work?")
  ```

## Streamlit

`streamlit run Home.py`

```python
import streamlit as st

st.title("Hello world!")

st.subheader('Welcome to streamlit')

st.markdown("""
            ### I love it
            """)

today = datetime.today().strftime("%H:%M:%S")
st.title(today)

model = st.selectbox(
    label="Choose model",
    options=(
        "GPT-3.5",
        "GPT-4",
    ),
)
if model == "GPT-3.5":
    st.write("cheap")
else:
    st.write("expensive")

name = st.text_input("What's your name?")
st.write(name)

value = st.slider(
    label="temperature",
    min_value=-5,
    max_value=40,
    step=5,
)
st.write(value)
```

## Document GPT

```python
import os
from typing import List
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
import streamlit as st

st.set_page_config(page_title="DocumentGPT", page_icon="📃")

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token: str, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
)

@st.cache_data(show_spinner=True)
def embed_file(file):
    file_content = file.read()
    if not os.path.exists("./.cache/files"):
        os.makedirs("./.cache/files")
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
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
    retriever = vectorstore.as_retriever()
    return retriever

def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )

def format_docs(docs: List[Document]):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Answer the question using ONLY the following context.
If you don't know the answer just say you don't know.
DON'T make anyting up.

Context: {context}
         """,
        ),
        ("human", "{question}"),
    ]
)

st.title("DocumentGPT")

st.markdown(
    """
Welcom!

Use the chatbot to ask questions to an AI about your files!

Upload your files on the sidebar.
"""
)

with st.sidebar:
    file = st.file_uploader(
        label="Upload a file(.txt, .pdf, .docs)",
        type=["pdf", "txt", "docx"],
    )

if file:
    retriever = embed_file(file)
    send_message("I'm ready Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about this file")
    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            response = chain.invoke(message)

else:
    st.session_state["messages"] = []
```

## HuggingFace

Huggingface에서 다양한 유료/무료 모델을 사용할 수 있다

[Mistral Model](https://huggingface.co/mistralai/Mistral-7B-v0.1), [API](https://huggingface.co/docs/api-inference/detailed_parameters#text-generation-task), [Docs](https://docs.mistral.ai/)

Instructor model을 사용하는 경우 아래의 가이드 참고

The template used to build a prompt for the Instruct model is defined as follows:

`<s>[INST] Instruction [/INST] Model answer</s>[INST] Follow-up instruction [/INST]`

```python
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    "[INST] What is the meaning of {word} [/INST]",
)

llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    model_kwargs={
        "max_new_tokens": 250,
    },
)

chain = prompt | llm

chain.invoke({"word": "flutter"})
```

## PrivateGPT

로컬에서 모델 실행

- PromptTemplate
  비교적 작은 사이즈의 `gpt-2` 모델을 사용
  gpt-2는 auto complete에 강점이 있다.

  ```python
  from langchain.llms.huggingface_pipeline import HuggingFacePipeline
  from langchain.prompts import PromptTemplate

  prompt = PromptTemplate.from_template(
      "A {word} is a",
  )

  llm = HuggingFacePipeline.from_model_id(
      model_id="gpt2",
      task="text-generation",
      device=-1,  # 0[GPU], -1[CPU(Default)]
      pipeline_kwargs={
          "max_new_tokens": 150,
      },
  )

  chain = prompt | llm

  chain.invoke({"word": "kimchi"})
  ```

- [GPT4ALL](https://gpt4all.io/index.html)
  - fine tunning 된 모델들을 직접 모델을 다운받아 사용할 수 있다
- [Ollama](https://ollama.ai/)

  - `ollama run {model}` 설치 및 실행
  - 모델 교체만으로 동일한 prompt를 실행 시킬 수 있다.
  - `OllamaEmbeddings`, `ChatOllama`

  ```python
  import os
  from typing import List
  from langchain.callbacks.base import BaseCallbackHandler
  from langchain.chat_models import ChatOllama
  from langchain.document_loaders import UnstructuredFileLoader
  from langchain.embeddings import CacheBackedEmbeddings, OllamaEmbeddings
  from langchain.prompts import ChatPromptTemplate
  from langchain.schema import Document
  from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
  from langchain.storage import LocalFileStore
  from langchain.text_splitter import CharacterTextSplitter
  from langchain.vectorstores.faiss import FAISS
  import streamlit as st

  st.set_page_config(page_title="PrivateGPT", page_icon="🔐")

  class ChatCallbackHandler(BaseCallbackHandler):
      message = ""

      def on_llm_start(self, *args, **kwargs):
          self.message_box = st.empty()

      def on_llm_end(self, *args, **kwargs):
          save_message(self.message, "ai")

      def on_llm_new_token(self, token: str, *args, **kwargs):
          self.message += token
          self.message_box.markdown(self.message)

  llm = ChatOllama(
      model="mistral:latest",
      temperature=0.1,
      callbacks=[
          ChatCallbackHandler(),
      ],
  )

  @st.cache_data(show_spinner=True)
  def embed_file(file):
      file_content = file.read()
      if not os.path.exists("./.cache/private_files"):
          os.makedirs("./.cache/private_files")
      file_path = f"./.cache/private_files/{file.name}"
      with open(file_path, "wb") as f:
          f.write(file_content)

      if not os.path.exists("./.cache/private_embeddings"):
          os.makedirs("./.cache/private_embeddings")
      cache_dir = LocalFileStore(f"./.cache/private_embeddings/{file.name}")
      text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
          separator="\n",
          chunk_size=600,
          chunk_overlap=100,
      )
      loader = UnstructuredFileLoader(file_path)
      documents = loader.load_and_split(text_splitter=text_splitter)
      embeddings = OllamaEmbeddings(model="mistral:latest")
      cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
          embeddings,
          cache_dir,
      )
      vectorstore = FAISS.from_documents(
          documents=documents,
          embedding=cached_embeddings,
      )
      retriever = vectorstore.as_retriever()
      return retriever

  def save_message(message, role):
      st.session_state["messages"].append({"message": message, "role": role})

  def send_message(message, role, save=True):
      with st.chat_message(role):
          st.markdown(message)
      if save:
          save_message(message, role)

  def paint_history():
      for message in st.session_state["messages"]:
          send_message(
              message["message"],
              message["role"],
              save=False,
          )

  def format_docs(docs: List[Document]):
      return "\n\n".join(doc.page_content for doc in docs)

  prompt = ChatPromptTemplate.from_template(
      """
  Answer the question using ONLY the following context and not training data.
  If you don't know the answer just say you don't know.
  DON'T make anyting up.

  Context: {context}
  Question: {question}
           """,
  )

  st.title("DocumentGPT")

  st.markdown(
      """
  Welcom!

  Use the chatbot to ask questions to an AI about your files!

  Upload your files on the sidebar.
  """
  )

  with st.sidebar:
      file = st.file_uploader(
          label="Upload a file(.txt, .pdf, .docs)",
          type=["pdf", "txt", "docx"],
      )

  if file:
      retriever = embed_file(file)
      send_message("I'm ready Ask away!", "ai", save=False)
      paint_history()
      message = st.chat_input("Ask anything about this file")
      if message:
          send_message(message, "human")
          chain = (
              {
                  "context": retriever | RunnableLambda(format_docs),
                  "question": RunnablePassthrough(),
              }
              | prompt
              | llm
          )
          with st.chat_message("ai"):
              response = chain.invoke(message)

  else:
      st.session_state["messages"] = []
  ```

## QuizGPT

- 주요 포인트: `Output Parser`, `Function Calling`
- [Wikipedia Api](https://python.langchain.com/docs/integrations/retrievers/wikipedia)
  ```python
  topic = st.text_input(label="Search Wikipedia...")
          if topic:
              retriever = WikipediaRetriever(top_k_results=5)  # type: ignore
              with st.status("Searching wikipedia..."):
                  docs = retriever.get_relevant_documents(topic)
                  st.write(docs)
  ```
- quiz를 생성하는 prompt → formatting prompt → output parser (json) → form ui

  ````python
  import os
  import json
  from typing import List
  from langchain.callbacks import StreamingStdOutCallbackHandler
  from langchain.chat_models import ChatOpenAI
  from langchain.document_loaders import UnstructuredFileLoader
  from langchain.prompts import ChatPromptTemplate
  from langchain.retrievers import WikipediaRetriever
  from langchain.schema import BaseOutputParser, Document
  from langchain.text_splitter import CharacterTextSplitter
  import streamlit as st

  def format_docs(docs: List[Document]):
      return "\n\n".join(doc.page_content for doc in docs)

  @st.cache_data(show_spinner="Loading file...")
  def split_file(file):
      file_content = file.read()
      if not os.path.exists("./.cache/quiz_files"):
          os.makedirs("./.cache/quiz_files")
      file_path = f"./.cache/quiz_files/{file.name}"
      with open(file_path, "wb") as f:
          f.write(file_content)

      text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
          separator="\n",
          chunk_size=600,
          chunk_overlap=100,
      )
      loader = UnstructuredFileLoader(file_path)
      documents = loader.load_and_split(text_splitter=text_splitter)
      return documents

  @st.cache_data(show_spinner="Making quiz...")
  # docs를 hashing할 수 없기 떄문에 topic으로 구분
  def run_quiz_chain(_docs, topic):
      chain = {"context": questions_chain} | formatting_chain | output_parser
      return chain.invoke(_docs)

  @st.cache_data(show_spinner="Searching wikipedia...")
  def wiki_search(term):
      retriever = WikipediaRetriever(
          top_k_results=5,
          # lang="ko",
      )  # type: ignore
      return retriever.get_relevant_documents(term)

  class JsonOutputParser(BaseOutputParser):
      def parse(self, text):
          text = text.replace("```json", "").replace("```", "")
          return json.loads(text)

  llm = ChatOpenAI(
      temperature=0.1,
      model="gpt-3.5-turbo-1106",
      streaming=True,
      callbacks=[StreamingStdOutCallbackHandler()],
  )

  questions_prompt = ChatPromptTemplate.from_messages(
      [
          (
              "system",
              """
  You are a helpful assistant that is role playing as a teacher.
  Based ONLY on the following context make 10 questions to test the user's knowledge about the text.
  Each question should have 4 answers, three of them must be incorrect and one should be correct.
  Use (o) to signal the correct answer.

  Question examples:

  Question: What is the color of the ocean?
  Answers: Red|Yellow|Green|Blue(o)

  Question: What is the capital or Georgia?
  Answers: Baku|Tbilisi(o)|Manila|Beirut

  Question: When was Avatar released?
  Answers: 2007|2001|2009(o)|1998

  Question: Who was Julius Caesar?
  Answers: A Roman Emperor(o)|Painter|Actor|Model

  Your turn!
  Make 10 different questions

  Context: {context}
  """,
          )
      ]
  )

  questions_chain = {"context": format_docs} | questions_prompt | llm

  formatting_prompt = ChatPromptTemplate.from_messages(
      [
          (
              "system",
              """
      You are a powerful formatting algorithm.

      You format exam questions into JSON format.
      Answers with (o) are the correct ones.

      Example Input:

      Question: What is the color of the ocean?
      Answers: Red|Yellow|Green|Blue(o)

      Question: What is the capital or Georgia?
      Answers: Baku|Tbilisi(o)|Manila|Beirut

      Question: When was Avatar released?
      Answers: 2007|2001|2009(o)|1998

      Question: Who was Julius Caesar?
      Answers: A Roman Emperor(o)|Painter|Actor|Model

      Example Output:

      ```json
      {{ "questions": [
              {{
                  "question": "What is the color of the ocean?",
                  "answers": [
                          {{
                              "answer": "Red",
                              "correct": false
                          }},
                          {{
                              "answer": "Yellow",
                              "correct": false
                          }},
                          {{
                              "answer": "Green",
                              "correct": false
                          }},
                          {{
                              "answer": "Blue",
                              "correct": true
                          }},
                  ]
              }},
                          {{
                  "question": "What is the capital or Georgia?",
                  "answers": [
                          {{
                              "answer": "Baku",
                              "correct": false
                          }},
                          {{
                              "answer": "Tbilisi",
                              "correct": true
                          }},
                          {{
                              "answer": "Manila",
                              "correct": false
                          }},
                          {{
                              "answer": "Beirut",
                              "correct": false
                          }},
                  ]
              }},
                          {{
                  "question": "When was Avatar released?",
                  "answers": [
                          {{
                              "answer": "2007",
                              "correct": false
                          }},
                          {{
                              "answer": "2001",
                              "correct": false
                          }},
                          {{
                              "answer": "2009",
                              "correct": true
                          }},
                          {{
                              "answer": "1998",
                              "correct": false
                          }},
                  ]
              }},
              {{
                  "question": "Who was Julius Caesar?",
                  "answers": [
                          {{
                              "answer": "A Roman Emperor",
                              "correct": true
                          }},
                          {{
                              "answer": "Painter",
                              "correct": false
                          }},
                          {{
                              "answer": "Actor",
                              "correct": false
                          }},
                          {{
                              "answer": "Model",
                              "correct": false
                          }},
                  ]
              }}
          ]
       }}
      ```
      Your turn!

      Questions: {context}

  """,
          )
      ]
  )

  formatting_chain = formatting_prompt | llm

  output_parser = JsonOutputParser()

  ################################################################################

  st.set_page_config(page_title="QuizGPT", page_icon="❓")

  st.title("QuizGPT")

  with st.sidebar:
      topic = None
      docs = None
      choice = st.selectbox(
          "Choose what you want to use",
          (
              "File",
              "Wikipedia Article",
          ),
      )

      if choice == "File":
          file = st.file_uploader(
              label="Upload a file(.txt, .pdf, .docs)",
              type=["pdf", "txt", "docx"],
          )
          if file:
              docs = split_file(file)

      else:
          topic = st.text_input(label="Search Wikipedia...")
          if topic:
              docs = wiki_search(topic)

  if not docs:
      st.markdown(
          """
  Welcom to QuizGPT.

  I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.

  Get started by uploading a file or searching on Wikipedia in the sidebar.

  """
      )
  else:
      response = run_quiz_chain(
          docs,
          topic if topic else file.name,  # type: ignore
      )
      st.write(response)
      with st.form(key="questions_form"):
          for question in response["questions"]:
              st.write(question["question"])
              value = st.radio(
                  key=question["question"],
                  label="Select correct answer",
                  options=[answer["answer"] for answer in question["answers"]],
                  index=None,
              )
              if {"answer": value, "correct": True} in question["answers"]:
                  st.success("✅")
              elif value is not None:
                  correct = list(
                      filter(lambda answer: answer["correct"], question["answers"])
                  )[0]["answer"]
                  st.error(f"❌: {correct}")

          button = st.form_submit_button()
  ````

- function calling (gpt)

  - function들을 제공하고 필요에 따라 사용하도록
  - 예시) 날씨 정보와 같은 실시간 정보는 함수를 통해 실시간 정보를 불러와 사용할 수 있다

  ```python
  import json
  from langchain.chat_models import ChatOpenAI
  from langchain.prompts import PromptTemplate

  def get_weather(longitude, latitude):
      print(f"weather for longitude({longitude}) and latitude({latitude})")

  function = {
      "name": "get_weather",
      "description": """
      function that takes longitude and latitude to find the weather of a place
      """,
      "parameters": {
          "type": "object",
          "properties": {
              "longitude": {
                  "type": "string",
                  "description": "The longitude coordinate",
              },
              "latitude": {
                  "type": "string",
                  "description": "The latitude coordinate",
              },
          },
      },
      "required": ["longitude", "latitude"],
  }

  llm = ChatOpenAI(temperature=0.1).bind(
      function_call="auto",
      functions=[function],
  )

  prompt = PromptTemplate.from_template("Who is the weather in {city}")

  chain = prompt | llm

  response = chain.invoke({"city": "rome"})

  response = response.additional_kwargs["function_call"]["arguments"]

  response = json.loads(response)

  get_weather(response["longitude"], response["latitude"])
  ```

## SiteGPT

TODOS

- [ ] Streaming
- [ ] Memory chat history
- [ ] Cache (similar) question
  - [ ] question list에서 유사한 질문을 찾는다
- AsyncChromiumLoader, Html2TextTransformer
- site를 load하고 html 문서를 text로 변환하는 과정

```python
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer
import streamlit as st

st.set_page_config(page_title="SiteGPT", page_icon="📊")

html2text_transformer = Html2TextTransformer()

################################################################
st.title("SiteGPT")

st.markdown(
    """
# SiteGPT

Ask questions about the content of a website.

Start by writing the URL of the website on the sidebar.
"""
)

with st.sidebar:
    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com",
    )

if url:
    loader = AsyncChromiumLoader(urls=[url])
    docs = loader.load()
    transformed = html2text_transformer.transform_documents(docs)
    st.write(transformed)
```

- sitemap loader
  - blog관련 url만 filter(regex)
  - parsing_function을 통해 불필요한 header, footer 등읱 요소 제거
  - text_splitter

```python
def parse_page(soup):
    """
    header와 footer를 제거한 contents 반환
    """
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")  # 줄바꿈
        .replace("\xa0", " ")  # 공백 문자
        .replace("CloseSearch Submit Blog", "")
    )

@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )

    loader = SitemapLoader(
        url,
        filter_urls=[r"^(.*\/blog\/).*"],
        parsing_function=parse_page,
    )
    loader.requests_per_second = 5  # 너무 빠르면 차단 당할 수 있다 (default 2)
    docs = loader.load_and_split(text_splitter=text_splitter)
    return docs
```

- Map Re Rank
  - 답변을 점수와 함께 제공하도록

```python
answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question.
    If you can't just say you don't know, don't make anything up.

    Then, give a score to the answer between 0 and 5.
    The score should be high if the answer is related to the user's question, and low otherwise.
    If there is no relevant content, the score is 0.
    Always provide scores with your answers
    Context: {context}

    Examples:

    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5

    Question: How far away is the sun?
    Answer: I don't know
    Score: 0

    Your turn!
    Question: {question}
"""
)

def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"context": doc.page_content, "question": question}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }

choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.
            Use the answers that have the highest score (more helpful) and favor the most recent ones.
            Cite sources and return the sources of the answers as they are with Date, do not change them.
            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)

def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f'{answer["answer"]}\nSource:{answer["source"]}\nDate{answer["date"]}\n'
        for answer in answers
    )

    return choose_chain.invoke(
        {"question": question, "answers": condensed},
    )
```

## MeetingGPT

영상 요약

비디오 → 오디오 추출(ffmpeg) → 10분 단위 chunk(split) → speech to text (whisper) → summarize whole content → embed

- 오디도 추출 (ffmpeg) -vn: ignore video
  `ffmpeg -i input_path -vn output_path`
  subprocess 로 cli를 실행할 수 있다
  ```python
  def extract_audio_from_video(video_path, audio_path):
      command = ["ffmpeg", "-i", video_path, "-vn", audio_path]
      subprocess.run(command)
  ```
- split audio (pydub)

```python
def split_audio(audio_path, chunks_folder, chunk_size=10):
    track = AudioSegment.from_mp3(audio_path)
    chunk_len = chunk_size * 60 * 1000
    chunks = math.ceil(len(track) / chunk_len)

    for i in range(chunks):
        start_time = i * chunk_len
        end_time = (i + 1) * chunk_len
        chunk = track[start_time:end_time]
        chunk.export(f"{chunks_folder}/chunk_{str(i).zfill(2)}.mp3", format="mp3")
```

- transcribe

```python
import openai
from glob import glob

def transcribe_chunks(chunk_folder, destination):
    files = sorted(glob(f"{chunk_folder}/*.mp3"))

    for file in files:
        with open(file, "rb") as audio_file, open(destination, "a") as text_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
            text_file.write(transcript["text"])

transcribe_chunks("./files/audios/chunks", "./files/audios/transcript.txt")
```

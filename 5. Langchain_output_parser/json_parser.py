from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="google/gemma-4-31B-it",
    task="text-generation"
    )

model=ChatHuggingFace(llm=llm)

parser=JsonOutputParser()

template1=PromptTemplate(
    template="""Give me the name age and city of a fictional person. \n {format_instructions}""",
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain=template1 | model | parser
result=chain.invoke({})
print(result)
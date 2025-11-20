from langchain_core.prompts import PromptTemplate
from langchain.chains import SequentialChain
from langchain.chains.llm import LLMChain

from langchain.chat_models import AzureChatOpenAI
import os

from dotenv import load_dotenv
load_dotenv()


llm = AzureChatOpenAI(
    deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    temperature=0.7
)


def generate_restaurant_name_and_items(cuisine):

    prompt_template_name = PromptTemplate(
        input_variables=['cuisine'],
        template=(
            "I want to open a modern and appealing restaurant serving {cuisine} cuisine. "
            "Suggest a unique, catchy, and respectful name suitable for a brand. "
            "Avoid culturally sensitive titles like royal ranks or religious references. "
            "Final output should only be a name of restaurant name."
            "example 'restaurant name: Curry Craft'."
        )
    )

    name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="restaurant_name")

    prompt_template_items = PromptTemplate(
        input_variables=['restaurant_name', 'cuisine'],
        template=(
            "The restaurant '{restaurant_name}' serves delicious {cuisine} cuisine. "
            "Suggest 6 to 8 authentic and popular menu items that match this cuisine. "
            "Include a mix of appetizers, mains, and sides if possible. "
            "Return only the names of the dishes as a comma-separated list, with no additional text and space or new line."
        )
    )

    food_items_chain = LLMChain(llm=llm, prompt=prompt_template_items, output_key="menu_items")

    chain = SequentialChain(
        chains=[name_chain, food_items_chain],
        input_variables=['cuisine'],
        output_variables=['restaurant_name', 'menu_items'],
        verbose=False
    )

    return chain.invoke({'cuisine': cuisine})

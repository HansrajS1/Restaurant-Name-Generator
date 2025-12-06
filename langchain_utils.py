from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI

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
        input_variables=["cuisine"],
        template=(
            "I want to open a modern, catchy, and brand-friendly restaurant serving {cuisine} cuisine. "
            "Suggest ONE unique restaurant name. "
            "Avoid culturally sensitive names like royal titles or religious references. "
            "Output ONLY the name. Example: 'Curry Craft'."
        )
    )

    name_chain = prompt_template_name | llm | StrOutputParser()

    prompt_template_items = PromptTemplate(
        input_variables=["restaurant_name", "cuisine"],
        template=(
            "The restaurant '{restaurant_name}' serves delicious {cuisine} cuisine. "
            "Suggest 6 to 8 authentic menu items. "
            "Return ONLY dish names, comma-separated, no extra text."
        )
    )

    food_items_chain = prompt_template_items | llm | StrOutputParser()

    full_chain = (
        {
            "restaurant_name": name_chain,
            "cuisine": RunnablePassthrough()
        }
        | food_items_chain
    )
    
    restaurant_name = name_chain.invoke({"cuisine": cuisine})
    menu_items = full_chain.invoke({"cuisine": cuisine})

    return {
        "restaurant_name": restaurant_name,
        "menu_items": menu_items
    }

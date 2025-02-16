import sys
sys.path.append("../")

from typing import Optional, Type
import ast
import unicodedata

from langchain_community.agent_toolkits.load_tools import load_tools
from pydantic import BaseModel, Field, field_validator
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from typing import List

from langchain_community.utilities.serpapi import SerpAPIWrapper

import pickle
import os
import pandas as pd


class CounterInput(BaseModel):
    in_str: str = Field(description="A List of lists composed in the form: [[sentence, character limit], [sentence, character limit],...]."\
                        "Sentence is the sentence to count the characters of, and character limit is the character limit the count should satisfy.")
    
    @field_validator('in_str', mode="before")
    def cast_to_string(cls, v):
        return str(v)

class CustomCounterTool(BaseTool):
    name: str = "character_counter"
    description: str = "A character counter. Useful for counting the number of characters in a sentence. Takes as input a List of lists composed in the form: [[sentence, character limit], [sentence, character limit],...]. \
        Sentence is the sentence to count the characters of, and character limit is the character limit the count should satisfy."
    args_schema: Type[BaseModel] = CounterInput
    return_direct: bool = False


    def _run(
        self, in_str: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> List[int]:
        in_sent = ast.literal_eval(in_str)
        """Returns the number of characters in each input sentence."""
        return_str = ""
        for sent in in_sent:
            c_count = count_chars(sent[0])
            limit = int(sent[1])
            return_str += f"{sent[0]}: {c_count}/{sent[1]} characters"
            if c_count > limit:
                return_str += " (Too long)\n"
            elif c_count < limit//2:
                return_str += " (Too short)\n"
            else:
                return_str += "\n"
        return return_str

def count_chars(s):
    count = 0
    for char in s:
        if unicodedata.east_asian_width(char) in ['F', 'W']:  # Full-width or Wide characters
            count += 2
        else:
            count += 1
    return count


class SerpAPIInput(BaseModel):
    in_str: str = Field(description="Input as String")

    @field_validator('in_str', mode="before")
    def cast_to_string(cls, v):
        return str(v)

class SerpAPITool(BaseTool):
    name: str = "google_search"
    description: str = "A search engine. Useful for when you need to answer questions about current events. Input should be a search query."
    args_schema: Type[BaseModel] = SerpAPIInput
    return_direct: bool = False

    def _run(
        self, in_str: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        searched_dict = retrieveSearches()
        if in_str in searched_dict:
            print("\nNote: Loaded from backup")
            return searched_dict[in_str]
        
        search_res = SerpAPIWrapper().run(in_str)
        searched_dict[in_str] = search_res
        saveSearches(searched_dict)
        return search_res



class OutputTool(BaseTool):
    name: str = "output_tool"
    description: str = "A tool to simply write your thoughts. Nothing will be return for output."
    args_schema: Type[BaseModel] = SerpAPIInput
    return_direct: bool = False
    handle_tool_error: bool = True

    def _run(
        self, in_str: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        return ""


class ClickAggregator(BaseTool):
    name: str = "click_aggregator"
    description: str = "Returns the total number of clicks per category for the current ad setting."
    args_schema: Type[BaseModel] = SerpAPIInput
    return_direct: bool = False
    click_df: pd.DataFrame = None

    def __init__(self, file):
        super().__init__()
        self.click_df = pd.read_csv(file)

    def _run(
        self, in_str: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        aggr_clicks = self.click_df.groupby("Category", as_index=False).sum()
        average = aggr_clicks["Clicks"].mean()
        return_text = ""
        for row in range(len(aggr_clicks)):
            return_text += f"Category: {aggr_clicks["Category"][row]}\nClicks: {aggr_clicks["Clicks"][row]}\nDifference to Average: {aggr_clicks["Clicks"][row] - average: .3f}\n\n"
        return return_text


from sympy import sympify, S, symbols,  Not, Or, And, Implies, Equivalent
import itertools


class MultStr(BaseModel):
    in_str: str = Field(description="Input as a list of strings, in the form of: [Expression1, Expression2, ...]")



class TruthTableGenerator(BaseTool):
    name: str = "truthtable_generator"
    description: str = "Returns the truth table for a given list of Boolean expression. Always use SymPy-style logical operators: \
                    And(A, B) for AND, Or(A, B) for OR, Not(A) for NOT, Implies(A, B) for IMPLIES, and Equivalent(A, B) for BICONDITIONAL. \
                    Parentheses can be used for grouping. Example: ['And(Or(Not(A), B), Implies(C, A))', ...]."

    args_schema: Type[BaseModel] = SerpAPIInput
    return_direct: bool = False

    def _run(
        self, in_str: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        if in_str[0] != "[":
            in_str = [in_str]
        else:
            in_str = ast.literal_eval(in_str)
        # print(in_str)
        try:
            expected_symbols = [
                "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
            ]
            local_dict = {name: symbols(name) for name in expected_symbols}
            local_dict.update({
                "~": Not,  # Negation
                "&": And,  # Logical AND
                "|": Or,   # Logical OR
                ">>": Implies,  # Logical implication
                "EQ": Equivalent  # Logical equivalence
            })

            # Parse the expressions
            parsed_expressions = [sympify(expr, locals=local_dict) for expr in in_str]
            if len(in_str) > 1:
                parsed_expressions.append(Equivalent(*parsed_expressions))
                in_str.append("Same Value for All Formulas")
            
            # Extract free symbols from all expressions
            free_syms = set().union(*(expr.free_symbols for expr in parsed_expressions))
            variables = sorted({str(s) for s in free_syms})

            # If there are no variables, each expression is constant
            if not variables:
                results = ["T" if expr else "F" for expr in parsed_expressions]
                return "Constant expressions:\n" + "\n".join(f"{expr} = {res}" for expr, res in zip(in_str, results))

            # Generate all possible truth assignments for the variables.
            truth_combinations = list(itertools.product([False, True], repeat=len(variables)))

            # Build the truth table rows.
            headers = " | ".join(variables) + " | " + " | ".join(in_str)
            separator = "-" * len(headers)
            table_rows = [headers, separator]

            same_val_list = []

            for values in truth_combinations:
                # Map variable names to boolean values.
                var_dict = dict(zip(variables, values))
                # Evaluate all expressions with these values.
                results = ["T" if expr.subs(var_dict) else "F" for expr in parsed_expressions]
                row_values = " | ".join("T" if v else "F" for v in values)
                table_rows.append(f"{row_values} | " + " | ".join(results))
                same_val_list.append(results[-1] == "T")
            
            if sum(same_val_list) == len(same_val_list):
                table_rows.append("\nIMPORTANT: THE GIVEN PROPOSITIONS ARE: **Logically Equivalent**")
                print("Logically Equivalent")
            elif sum(same_val_list) == 0:
                table_rows.append("\nIMPORTANT: THE GIVEN PROPOSITIONS ARE: **Contradictory**")
                print("Contradictory")


            return "\n".join(table_rows)

        except Exception as e:
            return f"Error processing expressions: {str(e)}"



class CounterexampleVerifier(BaseTool):
    name: str = "counterexample_verifier"
    description: str = "Verifies whether a given set of truth values serves as a counterexample to an argument. "\
                        "Input consists of premises, a conclusion, and a dictionary specifying truth values for variables, in the form of: "\
                        "{{\"premises\": [Premis1, ...], \"conclusion\": Conclusion, \"truth_values\": [{{variable1: \"True/False\", ...}}, ...]}}"\
                        "Uses SymPy-style logical operators: And(A, B), Or(A, B), Not(A), Implies(A, B), Equivalent(A, B). Make sure to give True False as a string."

    args_schema: Type[BaseModel] = SerpAPIInput
    return_direct: bool = False

    def _run(
        self, in_str: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        try:
            expected_symbols = [
                "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
            ]
            local_dict = {name: symbols(name) for name in expected_symbols}
            local_dict.update({
                "~": Not,  # Negation
                "&": And,  # Logical AND
                "|": Or,   # Logical OR
                ">>": Implies,  # Logical implication
                "EQ": Equivalent  # Logical equivalence
            })
            in_dict = ast.literal_eval(in_str)
            premises = in_dict["premises"]
            if len(premises) == 0:
                return "Empty premis, no evaluation possible."
            conclusion = in_dict["conclusion"]

            result = ""

            print(in_dict["truth_values"])

            for truth_value_list in in_dict["truth_values"]:
                truth_values = truth_value_list
                for key in truth_values.keys():
                    if truth_values[key] in ["True", "true", "T"]:
                        truth_values[key] = True
                    else:
                        truth_values[key] = False

                # Parse the premises and conclusion
                parsed_premises = [sympify(expr, locals=local_dict) for expr in premises]
                parsed_conclusion = sympify(conclusion, locals=local_dict)

                # Ensure the provided truth values match expected variables
                all_variables = set().union(*(expr.free_symbols for expr in parsed_premises + [parsed_conclusion]))
                var_dict = {str(var): truth_values[str(var)] for var in all_variables if str(var) in truth_values}

                # Evaluate premises and conclusion under the given truth assignment
                premises_results = [expr.subs(var_dict) for expr in parsed_premises]
                conclusion_result = parsed_conclusion.subs(var_dict)

                # Convert results to Boolean values
                premises_truths = [bool(result) for result in premises_results]
                conclusion_truth = bool(conclusion_result)

                # A counterexample occurs when all premises are true and the conclusion is false
                if all(premises_truths) and not conclusion_truth:
                    result += "For " + str(truth_value_list) + ": Valid counterexample. The given truth values make all premises true and the conclusion false.\n"
                else:
                    result += "For " + str(truth_value_list) + ": Not a counterexample.The given truth values do not satisfy the conditions for a counterexample.\n"
            return result
        except Exception as e:
            return f"Error processing expressions: {str(e)}"


import csv
class RejectWordTool(BaseTool):
    name: str = "reject_words"
    description: str = "A reject word checker. Checks whether each sentence contains words that should not be included. Takes as input a list composed in the form: [sentence1, sentence2, ...]."
    args_schema: Type[BaseModel] = CounterInput
    return_direct: bool = False
    reject_list: list = []

    def __init__(self, file_path):
        super().__init__()
        self.reject_list = []
        with open(file_path, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                self.reject_list += row


    def _run(
        self, in_str: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> List[int]:
        in_sent = ast.literal_eval(in_str)
        return_str = ""
        for sent in in_sent:
            rejected = []
            for reject in self.reject_list:
                if reject in sent:
                    rejected.append(reject)
            
            if len(rejected) > 0:
                return_str += f"{sent}: Rejected {rejected}\n"
            else:
                return_str += f"{sent}: Good\n"

        return return_str



def retrieveSearches():
    if not os.path.exists('../searches/searches.pkl'):
        return {}
    else:
        with open('../searches/searches.pkl', 'rb') as f:
            return pickle.load(f)

def saveSearches(key_dict):
    print("\nNote: Saved to backup")
    with open('../searches/searches.pkl', 'wb') as f:
        pickle.dump(key_dict, f)



def getSerpTool():
    # Comment out if live SerpAPI is needed
    return SerpAPITool()

    search_tool = load_tools(["serpapi"])
    search_tool[0].name = "google_search"
    return search_tool[0]




from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import TextLoader, CSVLoader
from okg.load_and_embed import customized_trend_retriever
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import Tool


def getTools(sel_tools, config):
    """ 
    0: SerpAPI
    1: Counter
    2,3: Ad Retriever
    4: Output
    5: Click Aggregator
    6: Python
    """
    agent_tools = []
    if 0 in sel_tools:
        agent_tools.append(getSerpTool())
    if 1 in sel_tools:
        agent_tools.append(CustomCounterTool())

    if 2 in sel_tools:
        KW_loader = CSVLoader(config["SETTING"]["initial_keyword_data"])
        KW_retriever = customized_trend_retriever(KW_loader, str(config['KEYS']['OPENAI_EMBEDDING_API_KEY']),  \
                                                  str(config['KEYS']['OPENAI_EMBEDDING_AZURE_OPENAI_ENDPOINT']))

        agent_tools.append(create_retriever_tool(
            KW_retriever,
            str(config['TOOL']['GOOD_KW_RETRIEVAL_NAME']),
            str(config['TOOL']['GOOD_KW_RETRIEVAL_DISCRPTION']),
        ))

    if 3 in sel_tools:
        exampler_loader = TextLoader(str(config['SETTING']['rule_data']))
        exampler_retriever = customized_trend_retriever(exampler_loader, str(config['KEYS']['OPENAI_EMBEDDING_API_KEY']),  \
                                                        str(config['KEYS']['OPENAI_EMBEDDING_AZURE_OPENAI_ENDPOINT'])) 

        agent_tools.append(create_retriever_tool(
            exampler_retriever,
            str(config['TOOL']['RULE_RETRIEVAL_NAME']),
            #'Search',
            str(config['TOOL']['RULE_RETRIEVAL_DISCRPTION']),
        ))
    
    if 4 in sel_tools:
        agent_tools.append(OutputTool())

    if 5 in sel_tools:
        agent_tools.append(ClickAggregator(config["SETTING"]["initial_keyword_data"]))

    if 6 in sel_tools:
        python_repl = PythonREPL()
        agent_tools.append(Tool(
            name="python_repl",
            description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
            func=python_repl.run,
        ))
    
    if 7 in sel_tools:
        agent_tools.append(RejectWordTool(config['TOOL']['REJECT_WORD_CSV']))
    
    if 8 in sel_tools:
        agent_tools.append(TruthTableGenerator())
    if 9 in sel_tools:
        agent_tools.append(CounterexampleVerifier())
    return agent_tools

